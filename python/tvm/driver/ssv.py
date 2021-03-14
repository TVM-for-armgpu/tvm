from tvm import tir
from tvm import te

DTYPE_SIZE_IN_WORD = {
    "float16": 0.5,
    "float32": 1,
}

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

class Report:
    def __init__(self, nthread, mx_fabric_size, mx_concur_fabric_size):
        self.nthread = nthread
        self.mx_fabric_size = mx_fabric_size
        self.mx_concur_fabric_size = mx_concur_fabric_size

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
            self.access_maps = []
            self.cur_access_map = []
            self.fabric_axis_map = {}
            self.nthread = 1

        def __call__(self, op):
            #print(f"found op {type(op)}")

            if isinstance(op, tir.BufferLoad):
                self.cur_access_map += [AccessFootPrint(op)]

            elif isinstance(op, tir.BufferStore):
                var_name = op.buffer.name

                self.cur_access_map = set(self.cur_access_map)

                self.access_maps += [(var_name, self.cur_access_map)]
                self.cur_access_map = []

            elif isinstance(op, tir.For):
                if op.kind == tir.ForKind.UNROLLED or op.kind == tir.ForKind.VECTORIZED:
                    self.fabric_axis_map[op.loop_var.name] = op.extent.value

            elif isinstance(op, tir.stmt.AttrStmt) and op.attr_key == "thread_extent":
                iter_var_name = op.node.var.name
                if iter_var_name.startswith("threadIdx"):
                    self.fabric_axis_map[iter_var_name] = op.value
                    self.nthread *= op.value

    v = Visitor()
    tir.stmt_functor.post_order_visit(f.body, v)

    # Calculate fabric sizing.
    mx_fabric_size = 0
    mx_concur_fabric_size = 0
    for out_var_name, input_var_access_fps in v.access_maps:
        total_fabric_size = 0
        total_concur_fabric_size = 0
        for access_fp in input_var_access_fps:
            # Aggregate fabric area.
            fabric_size = concur_fabric_size = DTYPE_SIZE_IN_WORD[access_fp.dtype]
            for idx in access_fp.idxs:
                extent = v.fabric_axis_map[idx] if idx in v.fabric_axis_map else 1

                if idx.startswith("threadIdx"):
                    concur_fabric_size *= extent
                else:
                    concur_fabric_size *= extent
                    fabric_size *= extent

            total_fabric_size += fabric_size
            if out_var_name != access_fp.name:
                total_concur_fabric_size += concur_fabric_size

        mx_fabric_size = max(mx_fabric_size, total_fabric_size)
        mx_concur_fabric_size = max(mx_concur_fabric_size, total_concur_fabric_size)

    #print(mx_fabric_size, mx_concur_fabric_size)

    return Report(v.nthread, mx_fabric_size, mx_concur_fabric_size)

def threading_score_higher_the_better(report, nthread_phys, nthread_logic):
    assert report.nthread >= nthread_phys // 2, f"under-used physical threads ({report.nthread} < {nthread_phys // 2})"
    assert report.nthread <= nthread_logic, f"thread count exceeds device limit ({report.nthread} > {nthread_logic})"
    return report.nthread

def memory_score_lower_the_better(report, mem_hierarchy):
    # Under-used register file.
    assert report.mx_fabric_size >= mem_hierarchy[0] // 4, f"under-use of registers ({report.mx_fabric_size} < {mem_hierarchy[0] // 4})"

    for i, size in enumerate(mem_hierarchy):
        if report.mx_concur_fabric_size <= size:
            return i
    raise AssertionError(f"fabric size exceeds device cache ({report.mx_concur_fabric_size} > {mem_hierarchy[-1]})")
    return len(mem_hierarchy)

# YOU CAN CHANGE THESE RULES
def validate_config(f, arch_detail):
    #print(f)
    # FIXME: Use `arch_detail` instead of hard-coded values.
    nthread_phys = 128
    nthread_logic = 1024
    mem_hierarchy = [128, 256, 16384]

    report = extract_access_map(f.body)
    try:
        threading_score = threading_score_higher_the_better(report, nthread_phys, nthread_logic)
        memory_score = memory_score_lower_the_better(report, mem_hierarchy)
        #print("threading_score=", threading_score, "memory_score=", memory_score)
    except Exception as e:
        raise AssertionError(f"SSV denied further compilattion of this config because of suboptimal resource usage: {e}")


class ArchDetail:
    def __init__(self, nthread_phys: int, nthread_logic: int, mem_hierarchy: list):
        self.nthread_phys = nthread_phys
        self.nthread_logic = nthread_logic
        self.mem_hierarchy = mem_hierarchy
