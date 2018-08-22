#pragma once

namespace DBX
{
	//deep copy of a unique pointer
	template< class T >
	std::unique_ptr<T> copy_unique(const std::unique_ptr<T>& source)
	{
		return source ? std::make_unique<T>(*source) : nullptr;
	};

	//deep copy of a shared pointer
	template< class T >
	std::shared_ptr<T> copy_shared(const std::shared_ptr<T>& source)
	{
		return source ? std::make_shared<T>(*source) : nullptr;
	};
}