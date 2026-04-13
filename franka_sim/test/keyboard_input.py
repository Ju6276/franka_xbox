import numpy as np
import pygame


class KeyboardInput:
    """
    Pygame keyboard input backend for the FR3 MuJoCo teleoperation demo.

    Keep the small pygame control window focused while teleoperating. This avoids
    sending the same keys to MuJoCo viewer shortcuts.
    """

    def __init__(self, position_step=0.01, rotation_step=0.05, gripper_step=0.1):
        pygame.init()
        pygame.display.set_caption("FR3 Keyboard Teleoperation")

        self.position_step = position_step
        self.rotation_step = rotation_step
        self.gripper_step = gripper_step
        self._exit_requested = False
        self._reset_requested = False
        self._gripper_command = 0.0
        self._screen = pygame.display.set_mode((520, 240))
        self._font = pygame.font.Font(None, 22)
        self._clock = pygame.time.Clock()

        self._print_controls()
        self._draw_controls()

    def _print_controls(self):
        print("Keyboard controls:")
        print("  Focus the 'FR3 Keyboard Teleoperation' pygame window.")
        print("  W/S or Up/Down: move +Y/-Y")
        print("  D/A or Right/Left: move +X/-X")
        print("  E/Q: move +Z/-Z")
        print("  L/J: roll +/-")
        print("  I/K: pitch +/-")
        print("  O/U: yaw +/-")
        print("  Z/X or [/]: open/close gripper")
        print("  C: stop gripper")
        print("  Space: reset environment")
        print("  Esc: exit")

    def _draw_controls(self):
        lines = [
            "Focus this pygame window for keyboard teleoperation.",
            "W/S or Up/Down: +Y/-Y    D/A or Right/Left: +X/-X",
            "E/Q: +Z/-Z",
            "L/J: roll +/-    I/K: pitch +/-    O/U: yaw +/-",
            "Z/X or [/]: open/close gripper    C: stop gripper",
            "Space: reset    Esc: exit",
        ]
        self._screen.fill((28, 30, 34))
        for i, line in enumerate(lines):
            surface = self._font.render(line, True, (235, 235, 235))
            self._screen.blit(surface, (18, 18 + i * 32))
        pygame.display.flip()

    def poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._exit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._exit_requested = True
                elif event.key == pygame.K_SPACE:
                    self._reset_requested = True
                elif event.key in (pygame.K_z, pygame.K_LEFTBRACKET):
                    self._gripper_command = -1.0
                elif event.key in (pygame.K_x, pygame.K_RIGHTBRACKET):
                    self._gripper_command = 1.0
                elif event.key == pygame.K_c:
                    self._gripper_command = 0.0

        self._clock.tick(60)

    def should_exit(self) -> bool:
        return self._exit_requested

    def consume_reset_requested(self) -> bool:
        reset_requested = self._reset_requested
        self._reset_requested = False
        return reset_requested

    def close(self):
        pygame.quit()

    def _axis(self, positive_keys, negative_keys):
        keys = pygame.key.get_pressed()
        positive = any(keys[key] for key in positive_keys)
        negative = any(keys[key] for key in negative_keys)
        return float(positive) - float(negative)

    def get_action(self):
        action = np.zeros(7)

        action[0] = self._axis((pygame.K_d, pygame.K_RIGHT), (pygame.K_a, pygame.K_LEFT))
        action[1] = self._axis((pygame.K_w, pygame.K_UP), (pygame.K_s, pygame.K_DOWN))
        action[2] = self._axis((pygame.K_e,), (pygame.K_q,))
        action[3] = self._axis((pygame.K_l,), (pygame.K_j,))
        action[4] = self._axis((pygame.K_i,), (pygame.K_k,))
        action[5] = self._axis((pygame.K_o,), (pygame.K_u,))

        action[6] = self._gripper_command

        action[:3] *= self.position_step
        action[3:6] *= self.rotation_step
        return action
