#include <arpa/inet.h>
#include <strings.h>
#include <unistd.h>
#include <fcntl.h>
#include <netdb.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#define MAX 80 
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
  
int main(int argc, char *argv[]) 
{ 
	if (argc != 3) {
        printf("usage:%s ip port\n", argv[0]);
        return 0;
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
        exit(0); 
    } else {
        printf("connected to the server..\n"); 
    }
  
    // function for chat 
    func(sockfd); 
  
    // close the socket 
    close(sockfd); 
} 

