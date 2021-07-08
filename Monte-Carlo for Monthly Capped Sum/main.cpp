#include <numeric>
#include <random>
#include <vector>
#include <algorithm>
#include <utility>
#include <iterator>

#include <execution>

#include <boost/math/distributions/normal.hpp>
#include <thrust/random/normal_distribution.h>
using Decimal = double;


class MarketData {
    Decimal annual_interest_;
    Decimal annual_vol_;
    Decimal time_step_;
    Decimal cap_rate_;

public:
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

    constexpr Decimal get_annual_interest() const {
        return annual_interest_;
    }

    constexpr Decimal get_annual_vol() const {
        return annual_vol_;
    }

    constexpr Decimal get_monthly_interest() const {
        return annual_interest_ * time_step_;
    }

    Decimal get_monthly_vol() const {
        return annual_vol_ * std::sqrt(time_step_);
    }

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

Decimal norm_inv(Decimal x, Decimal mean, Decimal stddev) {
    boost::math::normal n(mean, stddev);
    return boost::math::quantile(n, x);
}

Decimal growth_rate(Decimal x, Decimal mean, Decimal stddev) {
    return std::exp((mean - 0.5 * stddev * stddev) + stddev * x);
}

int main()
{
    const MarketData market{ 0.05, 0.1, 1.0 / 12.0, 0.05 };
    const Assumptions assumptions = { 3 * 12, 1000.0 };

    auto new_vec = [&]() {
        return allocate_vec_w_reserve(assumptions.term_months);
    };

    const auto policy = std::execution::par;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<Decimal> rand_vec = new_vec();

    std::generate_n(policy, std::back_inserter(rand_vec), assumptions.term_months, [&]() {
        return std::generate_canonical<Decimal, std::numeric_limits<Decimal>::digits>(gen);
        });

    std::vector<Decimal> norm_rand = new_vec();
    std::transform(policy, rand_vec.begin(), rand_vec.end(), std::back_inserter(norm_rand), [&](Decimal r) {
        return norm_inv(r, 0, 1);
        });

    std::normal_distribution<Decimal> nd(market.get_monthly_interest(), market.get_monthly_vol());
    

    std::vector<Decimal> growth_rate_vec = new_vec();
    std::transform(policy, norm_rand.begin(), norm_rand.end(), std::back_inserter(growth_rate_vec), [&](Decimal x) {
        return growth_rate(x, market.get_monthly_interest(), market.get_monthly_vol());
        });

    std::vector<Decimal> index = new_vec();
    std::inclusive_scan(policy, growth_rate_vec.begin(), growth_rate_vec.end(), std::back_inserter(index), std::multiplies<Decimal>{}, assumptions.index);

    std::vector<Decimal> pct_growth = new_vec();
    std::transform(policy, growth_rate_vec.begin(), growth_rate_vec.end(), std::back_inserter(pct_growth), [](Decimal x) { return x - 1; });

    std::vector<Decimal> capped_g = new_vec();
    std::transform(policy, pct_growth.begin(), pct_growth.end(), std::back_inserter(capped_g), [&](Decimal x) { return std::min(x, market.get_cap_rate()); });

    std::vector<Decimal> one_plus_capped_g = new_vec();
    std::transform(policy, capped_g.begin(), capped_g.end(), std::back_inserter(one_plus_capped_g), [&](Decimal x) { return x + 1; });

    Decimal pi_one_plus_capped_g = std::reduce(policy, one_plus_capped_g.begin(), one_plus_capped_g.end(), 1.0, std::multiplies<Decimal>{});
    Decimal monthly_cap_sum_payoff = std::max(1.0, pi_one_plus_capped_g);
    Decimal present_value_of_cash_flow = monthly_cap_sum_payoff * std::exp(-3.0 * market.get_annual_interest());


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