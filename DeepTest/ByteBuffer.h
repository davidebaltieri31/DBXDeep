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

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace DBX {
	class ByteBuffer {
	private:
		uint32_t rpos; //read position
		uint32_t wpos; //write position
		std::vector<uint8_t> buf; //the acutal buffer
		bool operation_failed = false; //whatever a  read/write operation has failed

		/* read a datum of templated type */
		template <typename T> T read() {
			T _data = read<T>(rpos);
			rpos += sizeof(T);
			return _data;
		}
		/* read a datum of templated type at position 'index'*/
		template <typename T> T read(uint32_t index) {
			if (index + sizeof(T) <= buf.size()) {
#ifdef __APPLE__
                unsigned char * _pBuf=&buf[index];
                T _fixValue;
                memcpy(&_fixValue, _pBuf, sizeof(T));
                return _fixValue;
#else
				return *((T*)&buf[index]);
#endif    
			}
			operation_failed = true; //read operation has failed
			return 0;
		}
		/* append templated datum at the end of buffer */
		template <typename T> void append(T data) {
			uint32_t _s = sizeof(data);
			if (size() < (wpos + _s)) {
				buf.resize(wpos + _s);
			}
			memcpy(&buf[wpos], (uint8_t*)&data, _s);
			wpos += _s;
		}
		/* insert templated datum at index */
		template <typename T> void insert(T data, uint32_t index) {
			if (index > size()) {
				operation_failed = true; //insert operation has failed
				return;
			}
			if ((index + sizeof(data)) > size()) {
				buf.resize(index + sizeof(data));
				wpos = index + sizeof(data);
			}
			int _sz = int(sizeof(data));
			for (int i = 0; i < _sz; ++i) {
				buf.insert(buf.begin() + index + i, *(&data + i));
			}
		}
		/* overwrite templated datum at index position */
		template <typename T> void write(T data, uint32_t index) {
			if ((index + sizeof(data)) > size()) {
				operation_failed = true; //insert operation has failed
				return;
			}
			memcpy(&buf[index], (uint8_t*)&data, sizeof(data));
			wpos = index + sizeof(data);
		}

	public:
		/* delete default constructors */
		ByteBuffer(const  ByteBuffer& other) = delete; //copy
		ByteBuffer& operator=(const ByteBuffer& other) = delete; //copy
		ByteBuffer(ByteBuffer&& other) = delete; //move
		ByteBuffer& operator=(ByteBuffer&& other) = delete; //move
		/* ByteBuffer constructor, Reserves specified size in internal vector */
		ByteBuffer(uint32_t size = 4096);
		/* ByteBuffer constructor, Consume an entire uint8_t array of length 'size' in the ByteBuffer. If arr==nullptr act as previous constructor */
		ByteBuffer(uint8_t* arr, uint32_t size);
		/* ByteBuffer constructor, Consume an entire std vector into the ByteBuffer. Uses vector swapping. input 'data' is de-facto cleared after this constructor */
		ByteBuffer(std::vector<unsigned char>& data);
		/* ByteBuffer Deconstructor */
		virtual ~ByteBuffer();
		/* Returns the number of bytes from the current read position till the end of the buffer */
		uint32_t bytesRemaining();
		/* Rewinds readpos */
		void rewind();
		/* Clears out all data from the internal vector (original preallocated size remains), resets the positions to 0 */
		void clear(bool force = false);
		/* Allocate an exact copy of the ByteBuffer on the heap and return a pointer with the exact same content */
		ByteBuffer* clone(bool copyState = false);
		void clone(ByteBuffer& into_this, bool copyState = false);
		/* Equals, test for data equivilancy, Compare this ByteBuffer to another by looking at each byte in the internal buffers and making sure they are the same */
		bool equals(ByteBuffer* other);
		bool equals(ByteBuffer& other);
		/* Reallocates memory for the internal buffer of size newSize. Read and write positions will also be reset */
		void resize(uint32_t newSize);
		void reserve(uint32_t newSize) { buf.reserve(newSize); }
		/* Returns the size of the internal buffer */
		uint32_t size();
		/* Returns pointer internal buffer data*/
		uint8_t* data();
		void swap(std::vector<uint8_t>& data);
		/* templated basic linear search */
		template <typename T> int32_t find(T key, uint32_t start = 0) {
			int32_t _ret = -1;
			uint32_t _len = uint32_t(buf.size());
			for (uint32_t _i = start; _i < _len; ++_i) {
				T _data = read<T>(_i);
				// Wasn't actually found, bounds of buffer were exceeded
				if ((key != 0) && (_data == 0) || operation_failed) {
					break;
				}
				// Key was found in array
				if (_data == key) {
					_ret = (int32_t)_i;
					break;
				}
			}
			return _ret;
		}
		/* Replacement occurences of 'key' with 'rep', seach from 'start'. If firstOccuranceOnly==true, replace the first occurance pnly. Otherwise replace all occurances. False by default */
		void replace(uint8_t key, uint8_t rep, uint32_t start = 0, bool firstOccuranceOnly = false);
		/* Collection of various read functions */
		/* Relative peek. Reads and returns the next uint8_t in the buffer from the current position but does not increment the read position */
		uint8_t peek();
		/* Relative get method. Reads the uint8_t at the buffers current position then increments the position */
		uint8_t get();
		/* Absolute get method. Read uint8_t at index (doesn't change internal read position)*/
		uint8_t get(uint32_t index); 
		/* Relative read into array buf of length len*/
		void getBytes(uint8_t* buf, uint32_t len);
		/* Relative read single byte*/
		uint8_t getByte();
		/* Absolute read single byte (doesn't change internal read position) */
		uint8_t getByte(uint32_t index); 
		/* Relative read single char*/
		char getChar(); // Relative
		/* Absolute read single char (doesn't change internal read position) */
		char getChar(uint32_t index);
		/* Relative read single double*/
		double getDouble();
		/* Absolute read single double (doesn't change internal read position) */
		double getDouble(uint32_t index);
		/* Relative read single float*/
		float getFloat();
		/* Absolute read single float (doesn't change internal read position) */
		float getFloat(uint32_t index);
		/* Relative read single unsigned int*/
		uint32_t getUnsignedInt();
		/* Absolute read single unsigned int (doesn't change internal read position) */
		uint32_t getUnsignedInt(uint32_t index);
		/* Relative read single unsigned long*/
		uint64_t getUnsignedLong();
		/* Absolute read single unsigned long (doesn't change internal read position) */
		uint64_t getUnsignedLong(uint32_t index);
		/* Relative read single unsigned short*/
		uint16_t getUnsignedShort();
		/* Absolute read single unsigned short (doesn't change internal read position) */
		uint16_t getUnsignedShort(uint32_t index);
		/* Relative read single signed int*/
		int32_t getSignedInt();
		/* Absolute read single signed int (doesn't change internal read position) */
		int32_t getSignedInt(uint32_t index);
		/* Relative read single signed short*/
		int16_t getSignedShort();
		/* Absolute read single signed short (doesn't change internal read position) */
		int16_t getSignedShort(uint32_t index);
		/* Relative read single signed long*/
		int64_t getSignedLong();
		/* Absolute read single signed long (doesn't change internal read position) */
		int64_t getSignedLong(uint32_t index);

		// Write
		/*-----------------------------Collection of various write functions-------------------------*/
		/* Relative write of the entire contents of another ByteBuffer (src) */
		void put(ByteBuffer* src); 
		void put(ByteBuffer& src);
		/* Relative write single uint8_t */
		void put(uint8_t b); 
		/* Absolute INSERT single uint8_t at index*/
		void put(uint8_t b, uint32_t index);
		/* Relative write byte array */
		void putBytes(uint8_t* b, uint32_t len);
		/* Absolute write byte array starting from index (overwrite data if insert == false)*/
		void putBytes(uint8_t* b, uint32_t len, uint32_t index, bool insert_ = false);
		/* Relative write single byte */
		void putByte(uint8_t value);
		/* Absolute write single byte at index*/
		void putByte(uint8_t value, uint32_t index);
		/* Relative write single char */
		void putChar(char value); 
		/* Absolute write single char at index*/
		void putChar(char value, uint32_t index); 
		/* Relative write single double */
		void putDouble(double value);
		/* Absolute write single double at index*/
		void putDouble(double value, uint32_t index);
		/* Relative write single float */
		void putFloat(float value);
		/* Absolute write single float at index*/
		void putFloat(float value, uint32_t index);
		/* Relative write single unsigned int */
		void putUnsignedInt(uint32_t value);
		/* Absolute write single unsigned int at index*/
		void putUnsignedInt(uint32_t value, uint32_t index);
		/* Relative write single signed int */
		void putSignedInt(int32_t value);
		/* Absolute write single signed int at index*/
		void putSignedInt(int32_t value, uint32_t index);
		/* Relative write single unsigned long */
		void putUnsignedLong(uint64_t value);
		/* Absolute write single unsigned long at index*/
		void putUnsignedLong(uint64_t value, uint32_t index);
		/* Relative write single signed long */
		void putSignedLong(uint64_t value);
		/* Absolute write single signed long at index*/
		void putSignedLong(uint64_t value, uint32_t index);
		/* Relative write single unsigned short */
		void putUnsignedShort(uint16_t value);
		/* Absolute write single unsigned short at index*/
		void putUnsignedShort(uint16_t value, uint32_t index);
		/* Relative write single signed short */
		void putSignedShort(int16_t value);
		/* Absolute write single signed short at index*/
		void putSignedShort(int16_t value, uint32_t index);
		/* Buffer Position Accessors & Mutators */
		/* skip bytes (read operations) */
		void skipBytes(uint32_t r);
		/* set reading position */
		void setReadPos(uint32_t r);
		/* get reading position */
		uint32_t getReadPos();
		/* set writing position */
		void setWritePos(uint32_t w);
		/* get writing position */
		uint32_t getWritePos();
		/* has a read/write operation failed */
		bool failed();
		/* reset failed_operation flag */
		void resetFailed();
		/* load buffer data from file */
		bool loadFromFile(std::string filename);
		/* save buffer data to file */
		bool saveIntoFile(std::string filename);
	};

}
