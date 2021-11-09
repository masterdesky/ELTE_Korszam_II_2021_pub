#include <iostream>

int addition(int &x) {

	int new_x = x + 1;

	return new_x;
}

int main(int argc, char const *argv[])
{
	// Starting number
	int x = 0;

	// Adding +1 to it 4 times
	for(int i = 0; i < 4; i++) {
		x = addition(x);
		std::cout << "Current value of `x` is " << x << std::endl;
	}

	return 0;
}