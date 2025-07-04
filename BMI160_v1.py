import board
import time
import busio
import math

# Constants
BMI160_I2C = 0x68
ACCEL_SENSITIVITY = 16384  # LSB/g at ±2g

# I2C setup
I2C = busio.I2C(board.GP1, board.GP0)
while not I2C.try_lock():
    pass

def setup_register(reg, data):
    I2C.writeto(BMI160_I2C, bytes([reg, data]))

def read_register(reg, length):
    I2C.writeto_then_readfrom(BMI160_I2C, bytes([reg]), result := bytearray(length))
    return result

def initialize_BMI160():
    setup_register(0x7E, 0x11)  # accel normal mode
    time.sleep(0.01)
    setup_register(0x7E, 0x15)  # gyro normal mode
    time.sleep(0.08)

def twos_complement(val, bits):
    if val & (1 << (bits - 1)):
        val -= 1 << bits
    return val

def read_raw_acceleration():
    data = read_register(0x12, 6)
    ax = twos_complement(int.from_bytes(data[0:2], 'little'), 16)
    ay = twos_complement(int.from_bytes(data[2:4], 'little'), 16)
    az = twos_complement(int.from_bytes(data[4:6], 'little'), 16)
    return ax, ay, az

def read_raw_gyroscope():
    data = read_register(0x0C, 6)
    gx = twos_complement(int.from_bytes(data[0:2], 'little'), 16)
    gy = twos_complement(int.from_bytes(data[2:4], 'little'), 16)
    gz = twos_complement(int.from_bytes(data[4:6], 'little'), 16)
    return gx, gy, gz

def read_gyro_range():
    data = read_register(0x43, 1)
    range_setting = data[0] & 0x07
    scales = {0x00: 16.4, 0x01: 32.8, 0x02: 65.5, 0x03: 131.0, 0x04: 262.4}
    return scales.get(range_setting, 131.0)

def auto_calibrate_accel():
    samples = 100
    ax_off = ay_off = az_off = 0
    for _ in range(samples):
        ax, ay, az = read_raw_acceleration()
        ax_off += ax
        ay_off += ay
        az_off += az
        time.sleep(0.005)
    ax_off //= samples
    ay_off //= samples
    az_off = (az_off // samples) - ACCEL_SENSITIVITY
    return ax_off, ay_off, az_off

def auto_calibrate_gyro():
    samples = 100
    gx_off = gy_off = gz_off = 0
    for _ in range(samples):
        gx, gy, gz = read_raw_gyroscope()
        gx_off += gx
        gy_off += gy
        gz_off += gz
        time.sleep(0.005)
    gx_off //= samples
    gy_off //= samples
    gz_off //= samples
    return gx_off, gy_off, gz_off

def read_acceleration(ax_off, ay_off, az_off):
    ax, ay, az = read_raw_acceleration()
    ax = ((ax - ax_off) / ACCEL_SENSITIVITY) * 9.81
    ay = ((ay - ay_off) / ACCEL_SENSITIVITY) * 9.81
    az = ((az - az_off) / ACCEL_SENSITIVITY) * 9.81
    return ax, ay, az

# Madgwick Filter (no numpy)
class Madgwick:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = [1.0, 0.0, 0.0, 0.0]

    def update(self, gx, gy, gz, ax, ay, az, dt):
        q1, q2, q3, q4 = self.q
        gx, gy, gz = [math.radians(val) for val in (gx, gy, gz)]

        norm = math.sqrt(ax**2 + ay**2 + az**2)
        if norm == 0:
            return
        ax /= norm
        ay /= norm
        az /= norm

        f1 = 2*(q2*q4 - q1*q3) - ax
        f2 = 2*(q1*q2 + q3*q4) - ay
        f3 = 2*(0.5 - q2**2 - q3**2) - az
        J = [
            [-2*q3,  2*q4, -2*q1, 2*q2],
            [ 2*q2,  2*q1,  2*q4, 2*q3],
            [ 0,    -4*q2, -4*q3, 0]
        ]
        step = [sum(J[j][i] * f for j, f in enumerate([f1, f2, f3])) for i in range(4)]
        norm_step = math.sqrt(sum(s**2 for s in step))
        step = [s / norm_step for s in step]

        q_dot = self._quat_multiply(self.q, [0, gx, gy, gz])
        q_dot = [0.5 * val - self.beta * step[i] for i, val in enumerate(q_dot)]
        self.q = [self.q[i] + q_dot[i] * dt for i in range(4)]
        norm_q = math.sqrt(sum(q**2 for q in self.q))
        self.q = [q / norm_q for q in self.q]

    def _quat_multiply(self, q, r):
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ]

    def get_euler(self):
        q1, q2, q3, q4 = self.q
        pitch = math.degrees(math.asin(2*(q1*q3 - q2*q4)))
        roll = math.degrees(math.atan2(2*(q1*q2 + q3*q4), 1 - 2*(q2**2 + q3**2)))
        yaw = math.degrees(math.atan2(2*(q1*q4 + q2*q3), 1 - 2*(q3**2 + q4**2)))
        return pitch, roll, yaw

# Initialization
initialize_BMI160()
gyro_scale = read_gyro_range()
ax_offset, ay_offset, az_offset = auto_calibrate_accel()
gx_offset, gy_offset, gz_offset = auto_calibrate_gyro()
filter = Madgwick(beta=0.1)

last_time = time.monotonic()

# Main loop
while True:
    ax, ay, az = read_acceleration(ax_offset, ay_offset, az_offset)
    gx_raw, gy_raw, gz_raw = read_raw_gyroscope()
    gx = (gx_raw - gx_offset) / gyro_scale
    gy = (gy_raw - gy_offset) / gyro_scale
    gz = (gz_raw - gz_offset) / gyro_scale

    current_time = time.monotonic()
    dt = current_time - last_time
    last_time = current_time

    filter.update(gx, gy, gz, ax, ay, az, dt)
    pitch, roll, yaw = filter.get_euler()

    print("Orientation:")
    print(f"  Pitch: {pitch:.2f}°")
    print(f"  Roll:  {roll:.2f}°")
    print(f"  Yaw:   {yaw:.2f}°")
    print("Acceleration (m/s²):")
    print(f"  ax: {ax:.2f}, ay: {ay:.2f}, az: {az:.2f}")
    print(f"Gyroscope (°/s):")
    print(f"  gx: {gx * gyro_scale:.2f}, gy: {gy * gyro_scale:.2f}, gz: {gz * gyro_scale:.2f}")
    print("-" * 40)

    time.sleep(0.2)# Write your code here :-)
