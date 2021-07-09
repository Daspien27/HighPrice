#include <numeric>
#include <random>
#include <vector>
#include <algorithm>
#include <utility>
#include <iterator>
#include <iostream>

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/complex.h>
#include <thrust/functional.h>

using Decimal = double;

class MarketData {
    Decimal annual_interest_;
    Decimal annual_vol_;
    Decimal time_step_;
    Decimal cap_rate_;

public:
    __host__ __device__
    constexpr MarketData(Decimal annual_interest,
        Decimal annual_vol,
        Decimal time_step,
        Decimal cap_rate) :
        annual_interest_(annual_interest),
        annual_vol_(annual_vol),
        time_step_(time_step),
        cap_rate_(cap_rate)
    {
    }

    __host__ __device__
    constexpr Decimal get_annual_interest() const {
        return annual_interest_;
    }

    __host__ __device__
    constexpr Decimal get_annual_vol() const {
        return annual_vol_;
    }

    __host__ __device__
    constexpr Decimal get_monthly_interest() const {
        return annual_interest_ * time_step_;
    }

    __host__ __device__
    Decimal get_monthly_vol() const {
        return annual_vol_ * std::sqrt(time_step_);
    }

    __host__ __device__
    constexpr Decimal get_cap_rate() const {
        return cap_rate_;
    }
};

struct Assumptions {
    int term_months;
    Decimal index;
};

std::vector<Decimal> allocate_vec_w_reserve(size_t length)
{
    std::vector<Decimal> v;
    v.reserve(length);
    return v;
}

//Decimal norm_inv(Decimal x, Decimal mean, Decimal stddev) {
//    boost::math::normal n(mean, stddev);
//    return boost::math::quantile(n, x);
//}

__host__ __device__
Decimal growth_rate(Decimal x, Decimal mean, Decimal stddev) {
    return std::exp((mean - 0.5 * stddev * stddev) + stddev * x);
}

struct get_random
{
    __host__
    get_random (MarketData const & market)
    : nrg {std::random_device () ()}
    , nd {market.get_monthly_interest(), market.get_monthly_vol()}
    {}

    std::random_device rd;
    std::mt19937_64 nrg;
    std::normal_distribution<Decimal> nd;

    Decimal operator () ()
    {
        return nd(nrg);
    }
};

std::vector<Decimal> RandomRates (size_t ScenariosCount, const MarketData & market, const Assumptions & assumptions)
{
    std::vector<Decimal> norm_rand;
    norm_rand.reserve (assumptions.term_months * ScenariosCount);

    std::generate_n(std::back_inserter(norm_rand), assumptions.term_months * ScenariosCount, get_random (market));

    return norm_rand;
}

Decimal ValueEstimateOnThrust (size_t ScenariosCount, const MarketData & market, const Assumptions & assumptions, thrust::device_vector<Decimal> const & norm_rand)
{
    thrust::device_vector<Decimal> scenarios (ScenariosCount, 1.0);
    for (auto i = 0; i < assumptions.term_months; ++i)
    {
        thrust::transform (scenarios.begin (), scenarios.end (), norm_rand.begin () + i * ScenariosCount, scenarios.begin (),
                            [market] __device__ (Decimal scenario, Decimal rate)
                            {
                                return scenario * (thrust::min (rate - 1, market.get_cap_rate ()) + 1.0);
                            });
    }

    thrust::transform (scenarios.begin (), scenarios.end (), scenarios.begin (),
                        [market] __device__ (auto rate)
                        {return thrust::max (1.0, rate) * thrust::exp(thrust::complex<Decimal>(-3.0 * market.get_annual_interest ())).real ();});
    
    return thrust::reduce (scenarios.begin (), scenarios.end (), 0.0, thrust::plus<Decimal> {}) / ScenariosCount;
}


Decimal ValueEstimateOnHost (size_t ScenariosCount, const MarketData & market, const Assumptions & assumptions, std::vector<Decimal> const & norm_rand)
{
    std::vector<Decimal> scenarios (ScenariosCount, 1.0);

    for (auto i = 0; i < assumptions.term_months; ++i)
    {
        std::transform (scenarios.begin (), scenarios.end (), norm_rand.begin () + i * ScenariosCount, scenarios.begin (),
                            [market] (Decimal scenario, Decimal rate)
                            {
                                return scenario * (std::min (rate - 1, market.get_cap_rate ()) + 1.0);
                            });
    }

    std::transform (scenarios.begin (), scenarios.end (), scenarios.begin (),
                        [market](auto rate) {return std::max (1.0, rate)* std::exp(-3.0 * market.get_annual_interest());});
    
    return std::reduce (scenarios.begin (), scenarios.end (), 0.0, std::plus<Decimal> {}) / ScenariosCount;
}


int main(int argc, char * argv[])
{
    ankerl::nanobench::Bench b;
    b.title("Vectors comparison");
    b.minEpochIterations(std::stoi(argv[2]));

    const size_t ScenariosCount = std::stoi (argv[1]);

    const MarketData market{ 0.05, 0.1, 1.0 / 12.0, 0.05 };
    const Assumptions assumptions = { 3 * 12, 1000.0 };

    auto norm_rand = RandomRates (ScenariosCount, market, assumptions); 
    thrust::device_vector<Decimal> norm_rand_device (norm_rand.size ());
    thrust::copy (norm_rand.begin (), norm_rand.end (), norm_rand_device.begin ());

    Decimal avgThrust, avgSTL;

    b.run("device", [&] () {avgThrust = ValueEstimateOnThrust (ScenariosCount, market, assumptions, norm_rand_device);});
    b.run("host",   [&] () {avgSTL = ValueEstimateOnHost (ScenariosCount, market, assumptions, norm_rand);});

    std::cout << "Device result:\t\t" << avgThrust << std::endl << "Host result:\t\t" << avgSTL << std::endl;

    return 0;
}