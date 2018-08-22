/* ===============================================================================
* Pikkart S.r.l. CONFIDENTIAL
* -------------------------------------------------------------------------------
* Copyright (c) 2016 Pikkart S.r.l. All Rights Reserved.
* Pikkart is a trademark of Pikkart S.r.l., registered in Europe,
* the United States and other countries.
*
* NOTICE:  All information contained herein is, and remains the property of
* Pikkart S.r.l. and its suppliers, if any. The intellectual and technical
* concepts contained herein are proprietary to Pikkart S.r.l. and its suppliers
* and may be covered by E.U., U.S. and Foreign Patents, patents in process,
* and are protected by trade secret or copyright law. Dissemination of this
* information or reproduction of this material is strictly forbidden unless
* prior written permission is obtained from Pikkart S.r.l..
* ===============================================================================*/


#include "ByteBuffer.h"
#include <fstream>

namespace DBX
{
	/* ByteBuffer constructor, Reserves specified size in internal vector */
	ByteBuffer::ByteBuffer(uint32_t size) {
		buf.reserve(size);
		clear();
	}
	/* ByteBuffer constructor, Consume an entire uint8_t array of length 'size' in the ByteBuffer. If arr==nullptr act as previous constructor */
	ByteBuffer::ByteBuffer(uint8_t* arr, uint32_t size) {
		// If the provided array is NULL, allocate a blank buffer of the provided size
		if (arr == nullptr) {
			buf.reserve(size);
			clear();
		}
		else { // Consume the provided array
			buf.reserve(size);
			clear();
			putBytes(arr, size);
		}
	}
	/* ByteBuffer constructor, Consume an entire std vector into the ByteBuffer. Uses vector swapping. data is de-facto cleared after this constructor */
	ByteBuffer::ByteBuffer(std::vector<unsigned char>& data) {
		clear();
		buf.swap(data);
		wpos += uint32_t(buf.size());
	}
	/* ByteBuffer Deconstructor */
	ByteBuffer::~ByteBuffer() {
		//clear();
	}
	/* Returns the number of bytes from the current read position till the end of the buffer */
	uint32_t ByteBuffer::bytesRemaining() {
		return size() - rpos;
	}
	/* Clears out all data from the internal vector (original preallocated size remains), resets the positions to 0 */
	void ByteBuffer::clear(bool force) {
		rpos = 0;
		wpos = 0;
		buf.clear();
		operation_failed = false;
		if (force)
		{
			std::vector<uint8_t>().swap(buf);
		}
	}
	/* Allocate an exact copy of the ByteBuffer on the heap and return a pointer with the exact same content */
	ByteBuffer* ByteBuffer::clone(bool copyState) {
		uint32_t _buf_size = uint32_t(buf.size());
		ByteBuffer* ret = new ByteBuffer(_buf_size); //FIXED
		memcpy(ret->buf.data(), buf.data(), sizeof(uint8_t)*_buf_size);
		if (copyState) {
			ret->rpos = rpos;
			ret->wpos = wpos;
		}
		else { // Reset positions
			ret->rpos = 0;
			ret->wpos = wpos;
		}
		return ret;
	}
	void ByteBuffer::clone(ByteBuffer& into_this, bool copyState) {
		uint32_t _buf_size = uint32_t(buf.size());
		into_this.resize(_buf_size);
		memcpy(into_this.buf.data(), buf.data(), sizeof(uint8_t)*_buf_size);
		if (copyState) {
			into_this.rpos = rpos;
			into_this.wpos = wpos;
		}
		else { // Reset positions
			into_this.rpos = 0;
			into_this.wpos = wpos;
		}
	}
	/* Equals, test for data equivilancy, Compare this ByteBuffer to another by looking at each byte in the internal buffers and making sure they are the same */
	bool ByteBuffer::equals(ByteBuffer* other) {
		// If sizes aren't equal, they can't be equal
		uint32_t _len = size();
		if (_len != other->size()) {
			return false;
		}
		// Compare byte by byte
		for (uint32_t _i = 0; _i < _len; _i++) {
			if ((uint8_t)get(_i) != (uint8_t)other->get(_i))
				return false;
		}
		return true;
	}
	bool ByteBuffer::equals(ByteBuffer& other) {
		// If sizes aren't equal, they can't be equal
		uint32_t _len = size();
		if (_len != other.size()) {
			return false;
		}
		// Compare byte by byte
		for (uint32_t _i = 0; _i < _len; _i++) {
			if ((uint8_t)get(_i) != (uint8_t)other.get(_i))
				return false;
		}
		return true;
	}
	/* Reallocates memory for the internal buffer of size newSize. Read and write positions will also be reset */
	void ByteBuffer::resize(uint32_t newSize) {
		buf.resize(newSize);
		rpos = 0;
		wpos = 0;
	}
	/* Rewinds readpos */
	void ByteBuffer::rewind() {
		rpos = 0;
		wpos = 0;
	}
	/* Returns the size of the internal buffer */
	uint32_t ByteBuffer::size() {
		return uint32_t(buf.size()*sizeof(uint8_t));
	}
	/* Returns pointer internal buffer data*/
	uint8_t* ByteBuffer::data() {
		return buf.data();
	}
	/* Replacement occurences of 'key' with 'rep', seach from 'start'. If firstOccuranceOnly==true, replace the first occurance pnly. Otherwise replace all occurances. False by default */
	void ByteBuffer::replace(uint8_t key, uint8_t rep, uint32_t start, bool firstOccuranceOnly) {
		uint32_t _len = uint32_t(buf.size());
		for (uint32_t _i = start; _i < _len; ++_i) {
			uint8_t _data = read<uint8_t>(_i);
			// Wasn't actually found, bounds of buffer were exceeded
			if ((key != 0) && (_data == 0) || operation_failed) {
				break;
			}
			// Key was found in array, perform replacement
			if (_data == key) {
				buf[_i] = rep;
				if (firstOccuranceOnly) {
					return;
				}
			}
		}
	}

	void ByteBuffer::swap(std::vector<uint8_t>& data)
	{
		clear();
		buf.swap(data);
		wpos += uint32_t(buf.size());
	}



	/*-----------------------------Collection of various read functions-------------------------*/
	/* Relative peek. Reads and returns the next uint8_t in the buffer from the current position but does not increment the read position */
	uint8_t ByteBuffer::peek() {
		return read<uint8_t>(rpos);
	}
	/* Relative get method. Reads the uint8_t at the buffers current position then increments the position */
	uint8_t ByteBuffer::get() {
		return read<uint8_t>();
	}
	/* Absolute get method. Read uint8_t at index (doesn't change internal read position) */
	uint8_t ByteBuffer::get(uint32_t index) {
		return read<uint8_t>(index);
	}
	/* Relative read into array buf of length len*/
	void ByteBuffer::getBytes(uint8_t* buf, uint32_t len) {
		for (uint32_t i = 0; i < len; i++) {
			buf[i] = read<uint8_t>();
		}
	}
	/* Relative read single byte*/
	uint8_t ByteBuffer::getByte() {
		return read<uint8_t>();
	}
	/* Absolute read single byte (doesn't change internal read position) */
	uint8_t ByteBuffer::getByte(uint32_t index) {
		return read<uint8_t>(index);
	}
	/* Relative read single char*/
	char ByteBuffer::getChar() {
		return read<char>();
	}
	/* Absolute read single char (doesn't change internal read position) */
	char ByteBuffer::getChar(uint32_t index) {
		return read<char>(index);
	}
	/* Relative read single double*/
	double ByteBuffer::getDouble() {
		return read<double>();
	}
	/* Absolute read single double (doesn't change internal read position) */
	double ByteBuffer::getDouble(uint32_t index) {
		return read<double>(index);
	}
	/* Relative read single float*/
	float ByteBuffer::getFloat() {
		return read<float>();
	}
	/* Absolute read single float (doesn't change internal read position) */
	float ByteBuffer::getFloat(uint32_t index) {
		return read<float>(index);
	}
	/* Relative read single unsigned int*/
	uint32_t ByteBuffer::getUnsignedInt() {
		return read<uint32_t>();
	}
	/* Absolute read single unsigned int (doesn't change internal read position) */
	uint32_t ByteBuffer::getUnsignedInt(uint32_t index) {
		return read<uint32_t>(index);
	}
	/* Relative read single signed int*/
	int32_t ByteBuffer::getSignedInt() {
		return read<int32_t>();
	}
	/* Absolute read single signed int (doesn't change internal read position) */
	int32_t ByteBuffer::getSignedInt(uint32_t index) {
		return read<int32_t>(index);
	}
	/* Relative read single unsigned long*/
	uint64_t ByteBuffer::getUnsignedLong() {
		return read<uint64_t>();
	}
	/* Absolute read single unsigned long (doesn't change internal read position) */
	uint64_t ByteBuffer::getUnsignedLong(uint32_t index) {
		return read<uint64_t>(index);
	}
	/* Relative read single unsigned long*/
	int64_t ByteBuffer::getSignedLong() {
		return read<int64_t>();
	}
	/* Absolute read single unsigned long (doesn't change internal read position) */
	int64_t ByteBuffer::getSignedLong(uint32_t index) {
		return read<int64_t>(index);
	}
	/* Relative read single unsigned short*/
	uint16_t ByteBuffer::getUnsignedShort() {
		return read<uint16_t>();
	}
	/* Absolute read single unsigned short (doesn't change internal read position) */
	uint16_t ByteBuffer::getUnsignedShort(uint32_t index) {
		return read<uint16_t>(index);
	}
	/* Relative read single signed short*/
	int16_t ByteBuffer::getSignedShort() {
		return read<int16_t>();
	}
	/* Absolute read single signed short (doesn't change internal read position) */
	int16_t ByteBuffer::getSignedShort(uint32_t index) {
		return read<int16_t>(index);
	}
	/*-----------------------------Collection of various write functions-------------------------*/
	/* Relative write of the entire contents of another ByteBuffer (src) */
	void ByteBuffer::put(ByteBuffer* src) {
		uint32_t len = src->size();
		for (uint32_t i = 0; i < len; i++)
			append<uint8_t>(src->get(i));
	}
	void ByteBuffer::put(ByteBuffer& src) {
		uint32_t len = src.size();
		for (uint32_t i = 0; i < len; i++)
			append<uint8_t>(src.get(i));
	}
	/* Relative write single uint8_t */
	void ByteBuffer::put(uint8_t b) {
		append<uint8_t>(b);
	}
	/* Absolute write single uint8_t at index*/
	void ByteBuffer::put(uint8_t b, uint32_t index) {
		insert<uint8_t>(b, index);
	}
	/* Relative write byte array*/
	void ByteBuffer::putBytes(uint8_t* b, uint32_t len) {
		// Insert the data one byte at a time into the internal buffer at position i+starting index
		for (uint32_t i = 0; i < len; i++) {
			append<uint8_t>(b[i]);
		}
	}
	/* Absolute write byte array starting from index (overwrite data)*/
	void ByteBuffer::putBytes(uint8_t* b, uint32_t len, uint32_t index, bool insert_) {
		// Insert the data one byte at a time into the internal buffer at position i+starting index
		if (insert_) {
			for (uint32_t i = 0; i < len; i++) {
				insert<uint8_t>(b[i], index++);
			}
		}
		else {
			wpos = index;
			// Insert the data one byte at a time into the internal buffer at position i+starting index
			for (uint32_t i = 0; i < len; i++) {
				append<uint8_t>(b[i]);
			}
		}
	}
	/* Relative write single byte */
	void ByteBuffer::putByte(uint8_t value) {
		append<uint8_t>(value);
	}
	/* Absolute write single byte at index*/
	void ByteBuffer::putByte(uint8_t value, uint32_t index) {
		write<uint8_t>(value, index);
	}
	/* Relative write single char */
	void ByteBuffer::putChar(char value) {
		append<char>(value);
	}
	/* Absolute write single char at index*/
	void ByteBuffer::putChar(char value, uint32_t index) {
		write<char>(value, index);
	}
	/* Relative write single double */
	void ByteBuffer::putDouble(double value) {
		append<double>(value);
	}
	/* Absolute write single double at index*/
	void ByteBuffer::putDouble(double value, uint32_t index) {
		write<double>(value, index);
	}
	/* Relative write single float */
	void ByteBuffer::putFloat(float value) {
		append<float>(value);
	}
	/* Absolute write single float at index*/
	void ByteBuffer::putFloat(float value, uint32_t index) {
		write<float>(value, index);
	}
	/* Relative write single unsigned int */
	void ByteBuffer::putUnsignedInt(uint32_t value) {
		append<uint32_t>(value);
	}
	/* Absolute write single unsigned int at index*/
	void ByteBuffer::putUnsignedInt(uint32_t value, uint32_t index) {
		write<uint32_t>(value, index);
	}
	/* Relative write single signed int */
	void ByteBuffer::putSignedInt(int32_t value) {
		append<int32_t>(value);
	}
	/* Absolute write single signed int at index*/
	void ByteBuffer::putSignedInt(int32_t value, uint32_t index) {
		write<int32_t>(value, index);
	}
	/* Relative write single unsigned long */
	void ByteBuffer::putUnsignedLong(uint64_t value) {
		append<uint64_t>(value);
	}
	/* Absolute write single unsigned long at index*/
	void ByteBuffer::putUnsignedLong(uint64_t value, uint32_t index) {
		write<uint64_t>(value, index);
	}
	/* Relative write single signed long */
	void ByteBuffer::putSignedLong(uint64_t value) {
		append<uint64_t>(value);
	}
	/* Absolute write single signed long at index*/
	void ByteBuffer::putSignedLong(uint64_t value, uint32_t index) {
		write<uint64_t>(value, index);
	}
	/* Relative write single unsigned short */
	void ByteBuffer::putUnsignedShort(uint16_t value) {
		append<uint16_t>(value);
	}
	/* Absolute write single unsigned short at index*/
	void ByteBuffer::putUnsignedShort(uint16_t value, uint32_t index) {
		write<uint16_t>(value, index);
	}
	/* Relative write single signed short */
	void ByteBuffer::putSignedShort(int16_t value) {
		append<int16_t>(value);
	}
	/* Absolute write single signed short at index*/
	void ByteBuffer::putSignedShort(int16_t value, uint32_t index) {
		write<int16_t>(value, index);
	}
	/* Buffer Position Accessors & Mutators */
	/* skip bytes (read operations) */
	void ByteBuffer::skipBytes(uint32_t r) {
		rpos += r;
	}
	/* set reading position */
	void ByteBuffer::setReadPos(uint32_t r) {
		rpos = r;
	}
	/* get reading position */
	uint32_t ByteBuffer::getReadPos() {
		return rpos;
	}
	/* set writing position */
	void ByteBuffer::setWritePos(uint32_t w) {
		wpos = w;
	}
	/* get writing position */
	uint32_t ByteBuffer::getWritePos() {
		return wpos;
	}
	/* has a read/write operation failed?*/
	bool ByteBuffer::failed() {
		return operation_failed;
	}
	/* reset failed_operation flag */
	void ByteBuffer::resetFailed() {
		operation_failed = false;
	}
	/* load buffer data from file */
	bool ByteBuffer::loadFromFile(std::string filename)
	{
		std::ifstream _file(filename, std::ios::binary);
		if (_file.is_open() == false) return false;
		_file.seekg(0, std::ios::end);    // go to the end
		int _length = int(_file.tellg());           // report location (this is the length)
		_file.seekg(0, std::ios::beg);    // go back to the beginning
		std::vector<uint8_t> _vector_buffer(_length);   // allocate memory for a buffer of appropriate dimension
		_file.read((char*)_vector_buffer.data(), _length);       // read the whole file into the buffer
		_file.close();                    // close file handle
		this->putBytes(_vector_buffer.data(), _length);
		return true;
	}
	/* save buffer data to file */
	bool ByteBuffer::saveIntoFile(std::string filename)
	{
		std::ofstream _file(filename, std::ios::binary);
		if (_file.is_open() == false) return false;
		_file.write((char*)this->data(), this->size());
		_file.close();
		return true;
	}
}