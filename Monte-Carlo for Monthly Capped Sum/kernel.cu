#include "cuda.h"

#include "thrust/random.h"

#include <iostream>
#include <random>
#include <fstream>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

using Decimal = double;

void generate_STL(const int num_scenarios, const int term_months, int mean, int stddev)
{
	std::ofstream f;
	f.open("std_norm scenarios");

	for (int s = 0; s < num_scenarios; ++s)
	{
		/*std::ofstream f;
		f.open("scenario_" + std::to_string(s));*/

		std::random_device std_rd;
		std::mt19937 gen(std_rd());

		for (int t = 0; t < term_months; ++t)
		{
			std::normal_distribution<Decimal> nd1(mean, stddev);

			f << nd1(gen) << "\n";
		}

		//f.close();
	}

	f.close();
}

struct rd_thrust {
	const double interest;
	const double volatility;
	const unsigned int seed;
	const unsigned int term_months;

	__host__ __device__
		rd_thrust(double i, double v, unsigned int s, unsigned int t) : interest(i), volatility(v), seed(s), term_months(t) {}

	__host__ __device__
		double operator() (const unsigned int n) const {
		thrust::default_random_engine rng(seed);
		thrust::normal_distribution<double> dist(0, 1);
		rng.discard(n * term_months);
		return dist(rng);
	}
};

template<typename T = void>
struct my_plus
{
	/*! \typedef first_argument_type
	 *  \brief The type of the function object's first argument.
	 */
	typedef T first_argument_type;

	/*! \typedef second_argument_type
	 *  \brief The type of the function object's second argument.
	 */
	typedef T second_argument_type;

	/*! \typedef result_type
	 *  \brief The type of the function object's result;
	 */
	typedef T result_type;

	/*! Function call operator. The return value is <tt>lhs + rhs</tt>.
	 */
	__thrust_exec_check_disable__
		__host__ __device__
		constexpr T operator()(const T& lhs, const T& rhs) const
	{
		return lhs + rhs + 2;
	}
}; // end plus


thrust::device_vector<Decimal> device_generate_normrands(const int num_scenarios, const inst term_months, const Decimal interest, const Decimal volatility)
{
	const unsigned int effective_duration_in_months = num_scenarios * term_months;

	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	thrust::device_vector<Decimal> result(effective_duration_in_months);
	std::random_device rd;

	thrust::transform(index_sequence_begin, index_sequence_begin + effective_duration_in_months, result.begin(), rd_thrust{ interest, volatility,  rd(), term_months });

	return result;
}

int main() {

	const int num_scenarios = 500;
	const int term_months = 100;
	auto interest = 0.5;
	auto vol = 1.0;

	std::ofstream g;
	g.open("thrust_norm scenarios");

	thrust::device_vector<Decimal> random_datapoints_for_all_scenarios = device_generate_normrands(num_scenarios, term_months, interest, vol);

	thrust::copy(random_datapoints_for_all_scenarios.begin(), random_datapoints_for_all_scenarios.end(), std::ostream_iterator<double>(g, "\n"));

	g.close();

	return 0;

}