#include <iostream>
#include <stdio.h>
#include "cpu_bitmap.h"

#define DIM 1000

struct Complex
{
	float real;
	float img;

	Complex(float x, float y)
		: real(x), img(y) {}

	float Magnitude2() { return real * real + img * img; }

	Complex operator*(const Complex& other)
	{
		return Complex((real * other.real - img * other.img), real * other.img + img * other.real);
	}

	Complex operator+(const Complex& other)
	{
		return Complex(real + other.real, img + other.img);
	}
};

int32_t Julia(int32_t x, int32_t y)
{
	const float scale = 1.5f;
	float jx = scale * (float)(DIM/2 - x)/(DIM/2);
	float jy = scale * (float)(DIM/2 - y)/(DIM/2);

	Complex c(0.37f, 0.1f);
	Complex z(jx, jy);

	for (int i = 0; i < 200; i++)
	{
		z = z * z + c;
		if (z.Magnitude2() > 1000)
			return 0;
	}
	return 1;
}

void Kernel(uint8_t* ptr)
{
	for (int y = 0; y < DIM; y++)
	{
		for (int x = 0; x < DIM; x++)
		{
			int32_t offset = x + y * DIM;

			int32_t juliaValue = Julia(x, y);
			ptr[offset * 4 + 0] = 255 * juliaValue; // R
			ptr[offset * 4 + 1] = 0;				// G
			ptr[offset * 4 + 2] = 0;				// B
			ptr[offset * 4 + 3] = 255;				// A
		}

	}
}

int main()
{
	CPUBitmap bitmap(DIM, DIM);
	uint8_t* ptr = bitmap.get_ptr(); // returns pixel coordinates

	Kernel(ptr);
	bitmap.display_and_exit();

	return 0;
}
