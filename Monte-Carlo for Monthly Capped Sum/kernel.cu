          
#include "cuda.h"

#include "thrust/random.h"

int main() {
    thrust::default_random_engine rd;

    rd.discard(1);


}
