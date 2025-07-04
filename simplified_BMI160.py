import time
import board
import busio
import math
import bmi160 as BMI160 


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


i2c = busio.I2C(board.GP1, board.GP0)
bmi = BMI160.BMI160(i2c, address=0x68)
print("BMI160 initialized.")

filter = Madgwick(beta=0.1)
last_time = time.monotonic()


while True:
    accx, accy, accz = bmi.acceleration  # m/s²
    gyrox, gyroy, gyroz = bmi.gyro       # °/s

    current_time = time.monotonic()
    dt = current_time - last_time
    last_time = current_time

    filter.update(gyrox, gyroy, gyroz, accx, accy, accz, dt)
    pitch, roll, yaw = filter.get_euler()

    print("Orientation:")
    print(f"  Pitch: {pitch:.2f}°")
    print(f"  Roll:  {roll:.2f}°")
    print(f"  Yaw:   {yaw:.2f}°")
    print("Acceleration (m/s²):")
    print(f"  x: {accx:.2f}, y: {accy:.2f}, z: {accz:.2f}")
    print("Gyroscope (°/s):")
    print(f"  x: {gyrox:.2f}, y: {gyroy:.2f}, z: {gyroz:.2f}")
    print("-" * 40)

    time.sleep(0.4)


