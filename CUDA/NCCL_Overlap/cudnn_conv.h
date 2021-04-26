#pragma once

#include <memory>
#include "comm.h"

void run_cudnn(int rank);

void run_cudnn_nccl(int rank, std::unique_ptr<Comm>& comm);
