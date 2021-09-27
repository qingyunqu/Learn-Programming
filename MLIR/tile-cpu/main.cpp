#include <chrono>
#include <iostream>

extern "C" float matmul();

int main(int argc, char *argv[]) {
  std::chrono::time_point<std::chrono::system_clock> startTime =
      std::chrono::system_clock::now();
  float v = matmul();
  std::chrono::duration<double> elapsedTime =
      std::chrono::system_clock::now() - startTime;

  std::cout << "matmul result = " << v << ", time = " << elapsedTime.count()
            << " seconds." << std::endl;
  return 0;
}
