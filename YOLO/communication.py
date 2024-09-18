"""
Communication protocol between server and client (handshake).
"""

import socket
import numpy as np
import struct
from typing import Union


# def send_array(sock: socket.socket, array: np.ndarray | None) -> None:
def send_array(sock: socket.socket, array: Union[np.ndarray, None]) -> None:
    """
    Function to send a NumPy array through a socket connection.
    """
    if array is None:
        # Send 0 as the signal for termination
        # (we can check ndim == 0 in recv_array)
        sock.sendall(struct.pack("i", 0))
        return
    
    # Send the number of dimensions in the array as integer.
    # For example, if the array has a shape (3, 4, 5), ndim will be 3.
    # The struct.pack("i", ndim) converts the integer into a binary representation, 
    # and sock.sendall sends this binary data over the network.
    ndim = array.ndim
    sock.sendall(struct.pack("i", ndim))

    # Send the shape of the array as a sequence of integers in binary format. 
    # For example, if the array has a shape (3, 4, 5), 
    # the sizes of each dimension (3, 4, and 5) are sent in sequence.
    # This is done by packing the shape information into a binary format and sending it over the network.
    sock.sendall(struct.pack(f"{ndim}i", *array.shape))

    # Send the length of the dtype string as an integer.
    # Convert the NumPy dtype object to its string representation (e.g., '<f4' for float32).
    # Encode this string representation into bytes using UTF-8 encoding.
    # Calculate the length of this byte sequence. 
    # Pack this length as a 4-byte integer ("i") in binary format.
    # Send the packed length over the network. 
    # This informs the receiver of the number of bytes to expect for the dtype string.
    dtype_str_bytes = array.dtype.str.encode("utf-8")
    sock.sendall(struct.pack("i", len(dtype_str_bytes)))

    # Send the dtype string itself encoded as UTF-8 bytes.
    # After sending the length of the dtype string, the next step is to send the actual dtype string.
    # Convert the dtype string (e.g., '<f4' for float32) to UTF-8 encoded bytes using the encode method.
    # Send these bytes over the network. 
    # This allows the receiver to reconstruct the dtype of the array correctly.
    sock.sendall(dtype_str_bytes)

    # Send the actual array data as bytes.
    # Convert the NumPy array to a byte representation using the tobytes method.
    # This method flattens the array and serializes its data into a continuous block of bytes.
    # For example, an array with shape (2, 2) and dtype float32 will be converted into bytes that
    # represent its raw memory content in a flattened form. 
    # This byte stream is then sent over the network.
    sock.sendall(array.tobytes())


# def recv_data(sock: socket.socket, size: int) -> bytes | None:
def recv_data(sock: socket.socket, size: int) -> Union[bytes, None]:
    """
    Helper function to receive 'size' bytes from the socket.
    """
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))

        # If the server or client closes the connection or there is a network error,
        # sock.recv returns an empty byte string (b""). In such cases, the function 
        # returns None, indicating the connection is closed.
        
        if packet == b"":
            return None # Connection closed

        data += packet

    return data


# def recv_array(sock: socket.socket) -> np.ndarray | None:
def recv_array(sock: socket.socket) -> Union[np.ndarray, None]:
    """
    Function to receive a NumPy array from the socket connection.
    """
    # Recieve the number of dimensions
    ndim_bytes = recv_data(sock=sock, size=struct.calcsize("i"))
    if ndim_bytes is None:
        return None # Connection closed
    ndim = struct.unpack("i", ndim_bytes)[0]

    # Check for termination signal
    if ndim == 0:
        return None # Connection closed

    # Recieve the shape of the array
    shape_bytes = recv_data(sock=sock, size=struct.calcsize(f"{ndim}i"))
    if shape_bytes is None:
        return None # Connection closed
    shape = struct.unpack(f"{ndim}i", shape_bytes)

    # Receive the dtype string length
    dtype_str_len_bytes = recv_data(sock=sock, size=struct.calcsize("i"))
    if dtype_str_len_bytes is None:
        return None # Connection closed
    dtype_len = struct.unpack("i", dtype_str_len_bytes)[0]
    
    # Receive the dtype string
    dtype_str_bytes = recv_data(sock=sock, size=dtype_len)
    if dtype_str_bytes is None:
        return None  # Connection closed
    dtype_str = dtype_str_bytes.decode('utf-8')

    # Receive the actual data
    dtype = np.dtype(dtype_str)
    buffer_size = np.prod(shape) * dtype.itemsize
    data_bytes = recv_data(sock=sock, size=buffer_size)
    if data_bytes is None:
        return None  # Connection closed

    # Reconstruct the NumPy array
    array = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    
    return array
