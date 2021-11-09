#include <iostream>

#include <vector>
#include <future>
#include <thread>

int addition(int x)
{
	int new_x = x + 1;

	return new_x;
}

int main(int argc, char const *argv[])
{
	// Starting number
	int x = 0;

	// Parallelization
	std::vector<std::future<int>> future_vec;

	// Adding +1 to it 4 times
	for(int i = 0; i < 4; i++)
	{
		future_vec.push_back(std::async(std::launch::async, addition, x));
	}

	for(int i = 0; i < 4; i++)
	{
		auto new_x = future_vec[i].get();
		std::cout << "Current value of `x` is " << new_x << std::endl;
	}

	return 0;
}