#include "comm.h"
#include "check.h"

#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/unistd.h>

#include <cuda_runtime.h>

static void commSend(int fd, const void* buf, size_t size) {
    // send size and msg
    write(fd, &size, sizeof(size_t));
    write(fd, buf, size);
}

static void* commReceive(int fd) {
    size_t size;
    read(fd, &size, sizeof(size_t));
    char* bufstart = new char[size];
    char* buf = bufstart;
    ssize_t size_received = 0, size_to_receive = size;
    while (size_to_receive > 0) {
        ssize_t bytes_read = read(fd, buf, size);
        if (bytes_read < 0)
            fprintf(stderr, "read error\n");
        buf += bytes_read;
        size_to_receive -= bytes_read;
    }
    return bufstart;
}

static void activeConnect(int socket_fd, sockaddr_in addr) {
    while (true) {
        int status;
        status = connect(socket_fd, (const sockaddr*)&addr, sizeof(sockaddr));
        if (status == 0)
            break;
    }
}

Comm::Comm(int nrank, int rank, int device, const char* ip_addr,
           uint16_t port) {
    this->rank_ = rank;
    this->nrank_ = nrank;
    this->device_ = device;
    CUDACHECK(cudaSetDevice(device));
    CUDACHECK(cudaStreamCreate(&this->stream));

    sockaddr_in addr;
    addr.sin_addr.s_addr = inet_addr(ip_addr);
    addr.sin_port = htons(port);
    addr.sin_family = AF_INET;
    if (rank == 0) {
        // server side code
        const char* uid = this->getUniqueId();
        int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (bind(socket_fd, (sockaddr*)&addr, sizeof(sockaddr)) < 0) {
            fprintf(stderr, "bind error\n");
        }
        if (listen(socket_fd, nrank) < 0) {
            fprintf(stderr, "listen error\n");
        }
        int connection_remain = nrank - 1;
        while (connection_remain > 0) {
            sockaddr_in daddr;
            socklen_t length = sizeof(sockaddr);
            int remote_fd;
            if ((remote_fd = accept(socket_fd, (sockaddr*)&daddr, &length)) <
                0) {
                fprintf(stderr, "accept error\n");
            }
            commSend(remote_fd, uid, 128);
            connection_remain--;
            close(remote_fd);
        }
        close(socket_fd);
    } else {
        // client side code
        int socket_fd = socket(AF_INET, SOCK_STREAM, 0);

        activeConnect(socket_fd, addr);
        const char* uid = (const char*)commReceive(socket_fd);
        this->setUniqueId(uid);
        delete uid;
        close(socket_fd);
    }
    this->commInitRank();
    printf("Connection Built %d\n", rank);
}

Comm::~Comm() {
    CUDACHECK(cudaStreamDestroy(this->stream));
    NCCLCHECK(ncclCommDestroy(this->comm));
}

const char* Comm::getUniqueId() {
    NCCLCHECK(ncclGetUniqueId(&this->commid));
    return this->commid.internal;
}

void Comm::setUniqueId(const char* buffer) {
    memcpy(this->commid.internal, buffer, 128);
}

void Comm::commInitRank() {
    NCCLCHECK(ncclCommInitRank(&this->comm, this->nrank_, this->commid,
                               this->rank_));
}

void Comm::syncStream() {
    CUDACHECK(cudaStreamSynchronize(this->stream));
}

void Comm::send(void* data, size_t N, ncclDataType_t dtype, int peer) {
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(data, N, dtype, peer, this->comm, this->stream));
    NCCLCHECK(ncclGroupEnd());
}

void Comm::recv(void* data, size_t N, ncclDataType_t dtype, int peer) {
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclRecv(data, N, dtype, peer, this->comm, this->stream));
    NCCLCHECK(ncclGroupEnd());
}

void Comm::allReduce(void* data, size_t N, ncclDataType_t dtype, ncclRedOp_t op) {
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce(data, data, N, dtype, op, this->comm, this->stream));
    NCCLCHECK(ncclGroupEnd());
}

void Comm::eventRecord(cudaEvent_t event) {
    CUDACHECK(cudaEventRecord(event, this->stream));
}
