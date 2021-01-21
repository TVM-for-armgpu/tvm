#include <arpa/inet.h>
#include <strings.h>
#include <unistd.h>
#include <fcntl.h>
#include <netdb.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#define MAX 800 
#define SA struct sockaddr 
void func(int sockfd) 
{ 
    char buff[MAX]; 
    int n; 
    for (;;) { 
        bzero(buff, sizeof(buff)); 
        //printf("Enter the string : "); 
        n = 0; 
        while ((buff[n++] = getchar()) != '\n'); 
        write(sockfd, buff, sizeof(buff)); 
        bzero(buff, sizeof(buff)); 
        read(sockfd, buff, sizeof(buff)); 
        printf("From Server : %s", buff); 
        if ((strncmp(buff, "exit", 4)) == 0) { 
            printf("Client Exit...\n"); 
            break; 
        }
    }
} 
typedef char bool;
#define true 1;
#define false 0;
bool IsBigEndian(){
    union NUM{
        int a;
        char b;
    }num;
    num.a = 0x1234;
    if( num.b == 0x12 ){
        return true;
    }
    return false;
}
uint32_t to_little_endian(uint32_t value){
    if (!IsBigEndian()) {return value;}
    return (value & 0x000000FFU) << 24 | (value & 0x0000FF00U) << 8 | 
        (value & 0x00FF0000U) >> 8 | (value & 0xFF000000U) >> 24; 
}
uint32_t to_big_endian(uint32_t le) {
    return (le & 0xff) << 24 
            | (le & 0xff00) << 8 
            | (le & 0xff0000) >> 8 
            | (le >> 24) & 0xff;
}

#define assert(condition) {\
    if (!(condition)){\
        printf("%s check failed\n",#condition);\
        exit(1);\
    }\
}
int probe_server(int sockfd) {
    char buff[MAX];
    int32_t magic_num = 193137;
    magic_num = to_little_endian(magic_num);
    write(sockfd, &magic_num, sizeof(magic_num));
    read(sockfd, buff, sizeof(magic_num));
    int32_t *p_magic_resp = (int32_t *)buff;
    assert(*p_magic_resp == magic_num);
    char *data = "[6]";
    int32_t len = to_little_endian(strlen(data));
    write(sockfd, &len,sizeof(len));
    write(sockfd, data,len);

    
    read(sockfd, &len, sizeof(len));
    if (IsBigEndian()) {
        len = to_big_endian(len);
    }
    read(sockfd, buff, len);
    buff[len]=0;
    printf("%s\n",buff);
    if (len > 40) {
        return 0;
    }
    return -1;

}
  

int probe_local_port_assigned(int port) {
    int ret = 0;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, "0.0.0.0", &addr.sin_addr);
    if(bind(fd, (struct sockaddr *)(&addr), sizeof(struct sockaddr_in)) < 0)
    {
        printf("port %d has been used.\n", port);
        ret = 1;
    } else {
        printf("port %d is avaiable.\n", port);
    }
    close(fd);
    return ret;
}

int main(int argc, char *argv[]) 
{ 
    if (argc == 2) {
        int port = atoi(argv[1]);
        exit(probe_local_port_assigned(port));
    }
	if (argc < 3) {
        printf("usage:%s ip port\n", argv[0]);
        return 0;
	}
    int  probe_mode = 0;
    if (argc == 4) {
        probe_mode = 1;
    }
    char ip[18]= {0};
    int port = atoi(argv[2]);
    strcpy(ip, argv[1]);
    printf("conncting to %s:%d\n",ip, port);
    
    int sockfd, connfd; 
    struct sockaddr_in servaddr, cli; 
  
    // socket create and varification 
    sockfd = socket(AF_INET, SOCK_STREAM, 0); 
    if (sockfd == -1) { 
        //printf("socket creation failed...\n"); 
        perror("socket");
        exit(0); 
    } 
    bzero(&servaddr, sizeof(servaddr)); 
  
    // assign IP, PORT 
    servaddr.sin_family = AF_INET; 
    servaddr.sin_addr.s_addr = inet_addr(ip); 
    servaddr.sin_port = htons(port); 
  
    // connect the client socket to server socket 
    if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) { 
        //printf("connection with the server failed...\n"); 
        perror("connect");
        exit(1); 
    } else {
        printf("connected to the server..\n"); 
    }
  
    if (probe_mode) {
        exit(probe_server(sockfd));
    } else {
    // function for chat 
        func(sockfd); 
    }
    // close the socket 
    close(sockfd); 
} 

