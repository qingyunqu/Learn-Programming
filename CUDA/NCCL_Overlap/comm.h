#pragma once
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/unistd.h>
#include "check.h"

class Comm {
private:
    ncclUniqueId m_commid;
    ncclComm_t m_comm;
    cudaStream_t m_stream;
    int m_rank, m_nrank, m_device;

private:
    const char* getUniqueId() {
        NCCLCHECK(ncclGetUniqueId(&this->m_commid));
        return this->m_commid.internal;
    }
    void setUniqueId(const char* buffer) {
        memcpy(this->m_commid.internal, buffer, 128);
    }
    void commInitRank() {
        NCCLCHECK(ncclCommInitRank(&this->m_comm, this->m_nrank, this->m_commid,
                                   this->m_rank));
    }

    void socketSend(int fd, const void* buf, size_t size) {
        // send size and msg
        write(fd, &size, sizeof(size_t));
        write(fd, buf, size);
    }
    void* socketReceive(int fd) {
        size_t size;
        read(fd, &size, sizeof(size_t));
        char* bufstart = new char[size];
        char* buf = bufstart;
        ssize_t size_to_receive = size;
        while (size_to_receive > 0) {
            ssize_t bytes_read = read(fd, buf, size);
            if (bytes_read < 0) {
                fprintf(stderr, "read error\n");
            }
            buf += bytes_read;
            size_to_receive -= bytes_read;
        }
        return bufstart;
    }
    void socketConnect(int socket_fd, sockaddr_in addr) {
        while (true) {
            int status;
            status = connect(socket_fd, (const sockaddr*)&addr,
                             sizeof(sockaddr));
            if (status == 0)
                break;
        }
    }

public:
    Comm(int nrank, int rank, int device, const char* ip_addr,
         unsigned short port) {
        CUDACHECK(cudaSetDevice(device));
        int leastPriority, greatestPriority;
        CUDACHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDACHECK(cudaStreamCreateWithPriority(&this->m_stream, cudaStreamNonBlocking, greatestPriority));
        this->m_rank = rank;
        this->m_nrank = nrank;
        this->m_device = device;

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
                if ((remote_fd = accept(socket_fd, (sockaddr*)&daddr,
                                        &length)) < 0) {
                    fprintf(stderr, "accept error\n");
                }
                socketSend(remote_fd, uid, 128);
                connection_remain--;
                close(remote_fd);
            }
            close(socket_fd);
        } else {
            // client side code
            int socket_fd = socket(AF_INET, SOCK_STREAM, 0);

            socketConnect(socket_fd, addr);
            const char* uid = (const char*)socketReceive(socket_fd);
            this->setUniqueId(uid);
            delete uid;
            close(socket_fd);
        }
        this->commInitRank();
        printf("Connection Built %d\n", rank);
    }

    ~Comm() {
        CUDACHECK(cudaStreamDestroy(this->m_stream));
        NCCLCHECK(ncclCommDestroy(this->m_comm));
    }

    void send(void* data, size_t N, ncclDataType_t dtype, int peer) {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(data, N, dtype, peer, this->m_comm, this->m_stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void recv(void* data, size_t N, ncclDataType_t dtype, int peer) {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclRecv(data, N, dtype, peer, this->m_comm, this->m_stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void allReduce(void* data, size_t N, ncclDataType_t dtype, ncclRedOp_t op) {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclAllReduce(data, data, N, dtype, op, this->m_comm,
                                this->m_stream));
        NCCLCHECK(ncclGroupEnd());
    }

    cudaStream_t getStream() { return this->m_stream; }
};
