import pygame
import numpy as np


class XboxInput:
    """
    DS4 / gamepad input backend for the FR3 MuJoCo teleoperation demo.

    This class captures inputs from a PS4 DualShock 4 controller and maps them
    to 7D teleoperation actions for the Franka simulation:

        [x, y, z, roll, pitch, yaw, gripper]

    Mapping used in this project:
    - Left stick (axis 0 / 1): XY translation
    - L1 / R1 (button 4 / 5): Z up / down
    - L2 / R2 (axis 2 / 5): roll
    - Right stick vertical (axis 4): pitch
    - Right stick horizontal (axis 3): yaw
    - Square / Circle (button 3 / 1): gripper open / close
    - PS (button 10): reset / go home
    - Options (button 9): exit teleoperation
    """

    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self._exit_requested = False
        self._reset_requested = False

        self._screen = pygame.display.set_mode((620, 260))
        pygame.display.set_caption("FR3 DS4 Teleoperation")
        self._font = pygame.font.Font(None, 24)
        self._clock = pygame.time.Clock()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No controller detected. Please connect a controller and try again.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print(f"Controller connected: {self.joystick.get_name()}")
        self._print_controls()
        self._draw_controls()

    def _print_controls(self):
        print("DS4 controls:")
        print("  Left stick: XY translation")
        print("  L1 / R1: +Z / -Z")
        print("  L2 / R2: roll +/-")
        print("  Right stick vertical: pitch")
        print("  Right stick horizontal: yaw")
        print("  Square / Circle: open / close gripper")
        print("  PS: reset environment / go home")
        print("  Options: exit teleoperation")

    def _draw_controls(self):
        lines = [
            "DS4 Teleoperation (DualShock 4)",
            "Left stick: XY translation",
            "L1 / R1: +Z / -Z",
            "L2 / R2: roll +/-",
            "Right stick vertical: pitch    Right stick horizontal: yaw",
            "Square / Circle: open / close gripper",
            "PS: reset / go home    Options: exit",
        ]

        self._screen.fill((28, 30, 34))
        for i, line in enumerate(lines):
            surface = self._font.render(line, True, (235, 235, 235))
            self._screen.blit(surface, (18, 18 + i * 32))
        pygame.display.flip()

    def poll_events(self):
        """Process controller and window events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._exit_requested = True

            elif event.type == pygame.JOYBUTTONDOWN:
                # Options -> exit
                if event.button == 9:
                    self._exit_requested = True
                    print("Exit requested")

                # PS -> reset / home
                elif event.button == 10:
                    self._reset_requested = True
                    print("Reset requested -> Home")

        self._clock.tick(60)

    def should_exit(self) -> bool:
        """Return whether teleoperation should exit."""
        return self._exit_requested

    def consume_reset_requested(self) -> bool:
        """Consume and clear reset request."""
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def close(self):
        """Release pygame resources."""
        if hasattr(self, "joystick"):
            self.joystick.quit()
        pygame.quit()

    def apply_dead_zone(self, value: float, threshold: float = 0.2) -> float:
        """
        Apply dead-zone filtering to joystick input.
        """
        return value if abs(value) >= threshold else 0.0

    def preprocess_trigger(self, value: float) -> float:
        """
        Map trigger raw value from [-1, 1] to [0, 1].
        """
        return (value + 1.0) / 2.0

    def get_action(self):
        """
        Get controller input and map it to:
            [x, y, z, roll, pitch, yaw, gripper]

        Returns:
            np.ndarray: 7D teleoperation action
        """
        action = np.zeros(7, dtype=np.float32)

        # Left stick: XY
        action[0] = self.apply_dead_zone(self.joystick.get_axis(0))          # left stick X
        action[1] = -1.0 * self.apply_dead_zone(self.joystick.get_axis(1))   # left stick Y

        # L1 / R1: Z
        if self.joystick.get_button(4):      # L1
            action[2] = 1.0
        elif self.joystick.get_button(5):    # R1
            action[2] = -1.0

        # L2 / R2: roll
        l2 = self.preprocess_trigger(self.joystick.get_axis(2))
        r2 = self.preprocess_trigger(self.joystick.get_axis(5))
        l2 = 0.0 if l2 < 0.1 else l2
        r2 = 0.0 if r2 < 0.1 else r2
        action[3] = l2 - r2

        # Right stick: pitch / yaw
        action[4] = -self.apply_dead_zone(self.joystick.get_axis(4))         # right stick vertical -> pitch
        action[5] = -0.5 * self.apply_dead_zone(self.joystick.get_axis(3))   # right stick horizontal -> yaw

        # Square / Circle: gripper
        if self.joystick.get_button(3):      # Square
            action[6] = -0.1
        elif self.joystick.get_button(1):    # Circle
            action[6] = 0.1

        # Scale translation and rotation
        action[:3] *= 0.001
        action[3:6] *= 0.005
        # action[6] keeps original incremental gripper command

        return action