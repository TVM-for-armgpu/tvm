import tvm
from tvm import tir
from tvm import te

# Remember to returns a set.
def collect_idx_var_names(buf_op):
    def impl(op):
        if isinstance(op, tir.Var):
            #print(op.name)
            return [op.name]
        rv = []
        try:
            rv += impl(op.a)
        except Exception:
            pass
        try:
            rv += impl(op.b)
        except Exception:
            pass
        return rv

    is_buf_op = isinstance(buf_op, tir.BufferLoad) or isinstance(buf_op, tir.BufferStore)
    assert is_buf_op, "can only collect index variables from vars"
    rv = []
    for idx in  buf_op.indices:
        rv += impl(idx)
    return rv

def collect_data_fabric(access_map):
    THREAD_ID_NAMES = {
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
    }

    rv = []
    for out_var_name, input_vars in access_map:
        # Returns a set.
        def find_fabric_axes(idxs_list):
            # Collect the global intersection of input variable indexers.
            intersection = None
            for idxs in idxs_list:
                # Ignore empty entries, they might be cleared in previous
                # iteration, and those are always pipeline contituents.
                if len(idxs) == 0:
                    continue
                # Collect the intersection of iter variables.
                if intersection == None:
                    intersection = set(idxs)
                else:
                    intersection.intersection_update(idxs)
            # If there is no intersection, and `input_variables` is not empty
            # then it's a data fabric, because there exists orthogonal access
            # via multiple axes. Return the intersection, i.e., the fabric axes.
            if intersection == None or len(intersection) == 0:
                return set().union(idx for idxs in idxs_list for idx in idxs)

            # Remove the common axes.
            for idxs in idxs_list:
                idxs.difference_update(intersection)

            # Next iteration
            return find_fabric_axes(idxs_list)

        fabric_idx_var_names = find_fabric_axes([input_var.idxs for input_var in input_vars])
        fabric_idx_var_names.difference_update(THREAD_ID_NAMES)
        # Ignore data pipelines.
        if len(fabric_idx_var_names) == 0:
            continue

        rv += [(out_var_name, input_vars, fabric_idx_var_names)]

    return rv

def extract_access_map(f):
    class AccessFootPrint:
        def __init__(self, bufload_op):
            self.name = bufload_op.buffer.name
            self.idxs = set(collect_idx_var_names(bufload_op))
            self.dtype = bufload_op.buffer.dtype

        def __repr__(self):
            return f"{self.name}{self.idxs}: {self.dtype}"

    class Visitor:
        def __init__(self):
            self.access_map = []
            self.cur_access_map = []

        def __call__(self, op):
            #print(f"found op {type(op)}")

            if isinstance(op, tir.BufferLoad):
                self.cur_access_map += [AccessFootPrint(op)]

            elif isinstance(op, tir.BufferStore):
                var_name = op.buffer.name

                # Ignore reduction temporary for itself
                self.cur_access_map = set(fp for fp in self.cur_access_map if fp.name != var_name)

                self.access_map += [(var_name, self.cur_access_map)]
                self.cur_access_map = []

    v = Visitor()
    tir.stmt_functor.post_order_visit(f.body, v)
    return v.access_map

class Config:
    def __init__(self, s):
        bounds = te.schedule.InferBound(s)
        self.iter_var_map = {}
        for iter_var, rng in bounds.items():
            iter_var_name = iter_var.var.name if isinstance(iter_var.var, tir.Var) else iter_var.var
            extent = int(rng.extent)
            self.iter_var_map[iter_var_name] = extent

    def get(self, name):
        return self.iter_var_map[name]
    def get_or_default(self, name, default):
        x = None
        try:
            x = self.get(name)
        except Exception:
            return default
        if x == None:
            return default
        return x

def threading_score_higher_the_better(access_map, cfg, nthread_phys, nthread_logic):
    nthread = cfg.get_or_default("threadIdx.x", 1) * cfg.get_or_default("threadIdx.y", 1) * cfg.get_or_default("threadIdx.z", 1)
    if nthread < nthread_phys // 2:
        return 0
    if nthread > nthread_logic:
        return 0
    return nthread

def memory_score_lower_the_better(access_map, cfg, mem_hierarchy):
    fabrics = collect_data_fabric(access_map)

    mx_fabric_size = 0
    for output_var_name, input_vars, fabric_idx_var_names in fabrics:
        fabric_size = 0
        for input_var in input_vars:
            var_access_size = 1
            for idx in input_var.idxs:
                if idx in fabric_idx_var_names:
                    var_access_size *= cfg.get_or_default(idx, 1)
                    #print('idx', idx, "=", cfg.get_or_default(idx, 1))
            fabric_size += var_access_size
        mx_fabric_size = max(mx_fabric_size, fabric_size)

    for i, size in enumerate(mem_hierarchy):
        if mx_fabric_size < size:
            return i
    return len(mem_hierarchy)

# YOU CAN CHANGE THESE RULES
def validate_config(access_map, cfg, nthread_phys, nthread_logic, mem_hierarchy):
    threading_score = threading_score_higher_the_better(access_map, cfg, nthread_phys, nthread_logic)
    memory_score = memory_score_lower_the_better(access_map, cfg, mem_hierarchy)
    return threading_score > 0 and memory_score < 2



class ArchDetail:
    def __init__(nthread_phys: int, nthread_logic: int, mem_hierarchy: list):
        self.nthread_phys = nthread_phys
        self.nthread_logic = nthread_logic
        self.mem_hierarchy = mem_hierarchy

def validate_config_pass(opts: dict):
    arch_detail = opts["ssv.arch_detail"]
    nthread_phys = arch_detail.nthread_phys
    nthread_logic = arch_detail.nthread_logic
    mem_hierarchy = arch_detail.mem_hierarchy

    def inner(f, *_):
        access_map = extract_access_map(f)
        cfg = Config(s)
        assert validate_config(access_map, cfg, nthread_phys, nthread_logic, mem_hierarchy), "SSV denied further compilattion of this config"
        return f

    return tir.transform.prim_func_pass(inner, opt_level=0)
