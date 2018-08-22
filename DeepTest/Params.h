#pragma once
#include <unordered_map>


#include <iostream>
#include <utility>
#include <typeinfo>
#include <type_traits>
#include <string>

#define DBX_THREAD 8
#define DBX_CORES 4

template <size_t arg1, size_t ... others>
struct static_max;

template <size_t arg>
struct static_max<arg>
{
	static const size_t value = arg;
};

template <size_t arg1, size_t arg2, size_t ... others>
struct static_max<arg1, arg2, others...>
{
	static const size_t value = arg1 >= arg2 ? static_max<arg1, others...>::value :
		static_max<arg2, others...>::value;
};

template<typename... Ts>
struct variant_helper;

template<typename F, typename... Ts>
struct variant_helper<F, Ts...> {
	inline static void destroy(size_t id, void * data)
	{
		if (id == typeid(F).hash_code())
			reinterpret_cast<F*>(data)->~F();
		else
			variant_helper<Ts...>::destroy(id, data);
	}

	inline static void move(size_t old_t, void * old_v, void * new_v)
	{
		if (old_t == typeid(F).hash_code())
			new (new_v) F(std::move(*reinterpret_cast<F*>(old_v)));
		else
			variant_helper<Ts...>::move(old_t, old_v, new_v);
	}

	inline static void copy(size_t old_t, const void * old_v, void * new_v)
	{
		if (old_t == typeid(F).hash_code())
			new (new_v) F(*reinterpret_cast<const F*>(old_v));
		else
			variant_helper<Ts...>::copy(old_t, old_v, new_v);
	}
};

template<> struct variant_helper<> {
	inline static void destroy(size_t id, void * data) { }
	inline static void move(size_t old_t, void * old_v, void * new_v) { }
	inline static void copy(size_t old_t, const void * old_v, void * new_v) { }
};

template<typename... Ts>
struct variant {
private:
	static const size_t data_size = static_max<sizeof(Ts)...>::value;
	static const size_t data_align = static_max<alignof(Ts)...>::value;

	using data_t = typename std::aligned_storage<data_size, data_align>::type;

	using helper_t = variant_helper<Ts...>;

	static inline size_t invalid_type() {
		return typeid(void).hash_code();
	}

	size_t type_id;
	data_t data;
public:
	variant() : type_id(invalid_type()) {   }

	variant(const variant<Ts...>& old) : type_id(old.type_id)
	{
		helper_t::copy(old.type_id, &old.data, &data);
	}

	variant(variant<Ts...>&& old) : type_id(old.type_id)
	{
		helper_t::move(old.type_id, &old.data, &data);
	}

	// Serves as both the move and the copy asignment operator.
	variant<Ts...>& operator= (variant<Ts...> old)
	{
		std::swap(type_id, old.type_id);
		std::swap(data, old.data);

		return *this;
	}

	template<typename T>
	void is() {
		return (type_id == typeid(T).hash_code());
	}

	void valid() {
		return (type_id != invalid_type());
	}

	template<typename T, typename... Args>
	void set(Args&&... args)
	{
		// First we destroy the current contents    
		helper_t::destroy(type_id, &data);
		new (&data) T(std::forward<Args>(args)...);
		type_id = typeid(T).hash_code();
	}

	template<typename T>
	T& get()
	{
		// It is a dynamic_cast-like behaviour
		if (type_id == typeid(T).hash_code())
			return *reinterpret_cast<T*>(&data);
		else
			throw std::bad_cast();
	}

	~variant() {
		helper_t::destroy(type_id, &data);
	}
};

class Params
{
private:
	using param_data = variant<std::string, double, int64_t>;

	std::unordered_map<std::string, param_data> m_params;
public:
	void set_int_param(std::string param_name, int32_t val) {
		param_data d;
		d.set<int64_t>(int64_t(val));
		m_params.emplace(param_name, d);
	}
	void set_float_param(std::string param_name, float val) {
		param_data d;
		d.set<double>(double(val));
		m_params.emplace(param_name, d);
	}
	void set_int64_param(std::string param_name, int64_t val) {
		param_data d;
		d.set<int64_t>(val);
		m_params.emplace(param_name, d);
	}
	void set_double_param(std::string param_name, double val) {
		param_data d;
		d.set<double>(val);
		m_params.emplace(param_name, d);
	}
	void set_string_param(std::string param_name, std::string val) {
		param_data d;
		d.set<std::string>(val);
		m_params.emplace(param_name, d);
	}

	int32_t get_int_param(std::string param_name) {
		return int32_t(m_params[param_name].get<int64_t>());
	}
	int64_t get_int64_param(std::string param_name) {
		return m_params[param_name].get<int64_t>();
	}
	float get_float_param(std::string param_name) {
		return float(m_params[param_name].get<double>());
	}
	double get_double_param(std::string param_name) {
		return m_params[param_name].get<double>();
	}
	std::string get_string_param(std::string param_name) {
		return m_params[param_name].get<std::string>();
	}
};