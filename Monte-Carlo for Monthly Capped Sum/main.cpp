#include <numeric>
#include <random>
#include <vector>
#include <algorithm>
#include <utility>
#include <iterator>
#include <iostream>
//#include <execution>

//#include <boost/math/distributions/normal.hpp>
#include <thrust/transform.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/scan.h>
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
    get_random (MarketData const & market)
    : rd {}
    , nd {market.get_monthly_interest(), market.get_monthly_vol()}
    {}

    std::random_device rd;
    std::normal_distribution<Decimal> nd;

    Decimal operator () ()
    {
        return nd(rd);
    }
};

int main(int argc, char * argv[])
{
    const size_t ScenariosCount = 1000;
    const MarketData market{ 0.05, 0.1, 1.0 / 12.0, 0.05 };
    const Assumptions assumptions = { 3 * 12, 1000.0 };

    auto new_vec = [&]() {
        return allocate_vec_w_reserve(assumptions.term_months);
    };

    std::vector<Decimal> norm_rand = new_vec();

    std::generate_n(std::back_inserter(norm_rand), assumptions.term_months, get_random (market));

    thrust::device_vector<Decimal> norm_rand_device (norm_rand.size ());
    thrust::copy (norm_rand.begin (), norm_rand.end (), norm_rand_device.begin ());

    thrust::device_vector<Decimal> growth_rate_vec (norm_rand_device.size ());
    thrust::transform(norm_rand_device.begin(), norm_rand_device.end(), growth_rate_vec.begin (), [market](Decimal x) {
        return growth_rate(x, market.get_monthly_interest(), market.get_monthly_vol());
        });

    //thrust::device_vector<Decimal> index (growth_rate_vec.size ());
    //thrust::inclusive_scan(growth_rate_vec.begin(), growth_rate_vec.end(), index.begin (), std::multiplies<Decimal>{}, assumptions.index);

    //thrust::device_vector<Decimal> pct_growth (growth_rate_vec.size ());
    //thrust::transform(growth_rate_vec.begin(), growth_rate_vec.end(), pct_growth.begin (), [](Decimal x) { return x - 1; });

    //thrust::device_vector<Decimal> capped_g (pct_growth.size ());
    //thrust::transform(pct_growth.begin(), pct_growth.end(), capped_g.begin (), [market](Decimal x) { return thrust::min(x, market.get_cap_rate()); });

    //thrust::device_vector<Decimal> one_plus_capped_g (capped_g.size ());
    //thrust::transform(capped_g.begin(), capped_g.end(), one_plus_capped_g.begin (), [](Decimal x) { return x + 1; });

    thrust::device_vector<Decimal> one_plus_capped_g (growth_rate_vec.size ());
    thrust::transform(  growth_rate_vec.begin(), growth_rate_vec.end(), one_plus_capped_g.begin (),
                        [market](Decimal x)
                        {
                            return thrust::min (x - 1, market.get_cap_rate ()) + 1; 
                        });

    Decimal pi_one_plus_capped_g = thrust::reduce(one_plus_capped_g.begin(), one_plus_capped_g.end(), 1.0, thrust::multiplies<Decimal>{});
    Decimal monthly_cap_sum_payoff = std::max(1.0, pi_one_plus_capped_g);
    Decimal present_value_of_cash_flow = monthly_cap_sum_payoff * std::exp(-3.0 * market.get_annual_interest());


    std::cout << present_value_of_cash_flow << std::endl;

    // reduce(all present values) / num(all present values)

  //  0,1,2,3,.....,10000
  //      ->
  // //pvs
  //      [](auto i) {
  //      
  //      // generate i * assumptions.term_months
  //      // get normal dists
  //      // ???
  //      // profit
  //      //return pv
  //  }

    return 0;
}