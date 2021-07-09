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
#include <thrust/random.h>

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

__host__ __device__
Decimal growth_rate(Decimal x, Decimal mean, Decimal stddev) {
    return std::exp((mean - 0.5 * stddev * stddev) + stddev * x);
}

struct get_random
{
    __host__
    get_random (MarketData const & market)
    : nrg {std::random_device () ()}
    , nd {market.get_monthly_interest() - market.get_monthly_vol() * market.get_monthly_vol() / 2.0, market.get_monthly_vol()}
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


Decimal ValueEstimateOnThrustAlt    (   size_t ScenariosCount
                                    ,   const MarketData & market
                                    ,   const Assumptions & assumptions
                                    ,   thrust::device_vector<Decimal> const & norm_rand)
{
    thrust::device_vector<Decimal> scenarios (ScenariosCount, 1.0);
    thrust::counting_iterator<size_t> first (0);
    thrust::counting_iterator<size_t> last (ScenariosCount); 

    thrust::transform (first, last, scenarios.begin (),
                            [assumptions, market, norms = norm_rand.data ()] __device__ (size_t scenario_index)
                            {
                                auto offset = scenario_index * assumptions.term_months;
                                auto prod = 1.0;
                                for (auto n = norms + offset; n < norms + offset + assumptions.term_months; ++n)
                                {
                                    prod *= (thrust::min (std::exp(*n) - 1, market.get_cap_rate ()) + 1.0);
                                }
                                return thrust::max (prod, 1.0) * std::exp((-assumptions.term_months / 12.0) * market.get_annual_interest ());
                            });
    
    return thrust::reduce (scenarios.begin (), scenarios.end (), 0.0, thrust::plus<Decimal> {}) / ScenariosCount;
}


Decimal ValueEstimateOnThrust   (   size_t ScenariosCount
                                ,   const MarketData & market
                                ,   const Assumptions & assumptions
                                ,   thrust::device_vector<Decimal> const & norm_rand)
{
    thrust::device_vector<Decimal> scenarios (ScenariosCount, 1.0);
    for (auto i = 0; i < assumptions.term_months; ++i)
    {
        thrust::transform (scenarios.begin (), scenarios.end (), norm_rand.begin () + i * ScenariosCount, scenarios.begin (),
                            [market] __device__ (Decimal scenario, Decimal rate)
                            {
                                return scenario * (thrust::min (std::exp(rate) - 1, market.get_cap_rate ()) + 1.0);
                            });
    }

    thrust::transform (scenarios.begin (), scenarios.end (), scenarios.begin (),
                        [market, assumptions] __device__ (auto rate)
                        {return thrust::max (1.0, rate) * thrust::exp(thrust::complex<Decimal>((-assumptions.term_months / 12.0) * market.get_annual_interest ())).real ();});

    return thrust::reduce (scenarios.begin (), scenarios.end (), 0.0, thrust::plus<Decimal> {}) / ScenariosCount;
}


Decimal ValueEstimateOnHost (   size_t ScenariosCount
                            ,   const MarketData & market
                            ,   const Assumptions & assumptions
                            ,   std::vector<Decimal> const & norm_rand)
{
    std::vector<Decimal> scenarios (ScenariosCount, 1.0);

    for (auto i = 0; i < assumptions.term_months; ++i)
    {
        std::transform (scenarios.begin (), scenarios.end (), norm_rand.begin () + i * ScenariosCount, scenarios.begin (),
                            [market] (Decimal scenario, Decimal rate)
                            {
                                return scenario * (std::min (std::exp(rate) - 1, market.get_cap_rate ()) + 1.0);
                            });
    }

    std::transform (scenarios.begin (), scenarios.end (), scenarios.begin (),
                        [market, assumptions](auto rate) {return std::max (1.0, rate)* std::exp((-assumptions.term_months / 12.0) * market.get_annual_interest());});

    return std::reduce (scenarios.begin (), scenarios.end (), 0.0, std::plus<Decimal> {}) / ScenariosCount;
}

namespace random_device_generator
{

struct rd_thrust {
	const double interest;
	const double volatility;
	const unsigned int seed;
	const int term_months;

	__host__ __device__
		rd_thrust(double i, double v, unsigned int s, int t) : interest(i), volatility(v), seed(s), term_months(t) {}

	__host__ __device__
		double operator() (const unsigned int n) const {
		thrust::default_random_engine rng(seed);
		thrust::normal_distribution<double> dist(interest - volatility * volatility / 2.0, volatility);
		rng.discard(n * term_months);
		return dist(rng);
	}
};

thrust::device_vector<Decimal> device_generate_normrands(const size_t num_scenarios, const int term_months, const Decimal interest, const Decimal volatility)
{
	const unsigned int effective_duration_in_months = num_scenarios * term_months;

	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	thrust::device_vector<Decimal> result(effective_duration_in_months);
	std::random_device rd;

	thrust::transform(index_sequence_begin, index_sequence_begin + effective_duration_in_months, result.begin(), rd_thrust{ interest, volatility,  rd(), term_months });

	return result;
}

}

int main(int argc, char * argv[])
{
    ankerl::nanobench::Bench b;
    b.title("Vectors comparison");
    b.minEpochIterations(std::stoi(argv[2]));

    const size_t ScenariosCount = std::stoi (argv[1]);

    const MarketData market{ 0.05, 0.1, 1.0 / 12.0, 0.05 };
    const Assumptions assumptions = { 3 * 12, 1000.0 };

    std::vector<Decimal> norm_rand;
    b.run ("rng host",  [&] () {norm_rand = RandomRates (ScenariosCount, market, assumptions);});
    thrust::device_vector<Decimal> norm_rand_device;
    b.run ("rng device",    [&] () {norm_rand_device = random_device_generator::device_generate_normrands (ScenariosCount, assumptions.term_months, market.get_monthly_interest (), market.get_monthly_vol ());});

    Decimal avgThrustAlt, avgThrust, avgSTL;

    b.run("device alt", [&] () {avgThrustAlt = ValueEstimateOnThrustAlt (ScenariosCount, market, assumptions, norm_rand_device);});
    b.run("device",     [&] () {avgThrust = ValueEstimateOnThrust (ScenariosCount, market, assumptions, norm_rand_device);});
    b.run("host",       [&] () {avgSTL = ValueEstimateOnHost (ScenariosCount, market, assumptions, norm_rand);});

    std::cout << "Device result alt:\t" << avgThrustAlt << '\n' << "Device result:\t\t" << avgThrust << std::endl << "Host result:\t\t" << avgSTL << std::endl;

    return 0;
}