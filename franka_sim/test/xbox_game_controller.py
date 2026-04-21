import pygame


# 初始化 Pygame 和 Joystick
pygame.init()
pygame.joystick.init()

# 检查是否有手柄连接
if pygame.joystick.get_count() == 0:
    print("No joystick detected. Please connect a joystick and try again.")
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Attached joystick: {joystick.get_name()}")

# 循环检测手柄输入
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN and event.button == 11:
            running = False 
            print("Exit")
        
        # 检测按键按下
        if event.type == pygame.JOYBUTTONDOWN:
            print(f"Button {event.button} pressed")

        # 检测摇杆移动
        def apply_dead_zone(value, threshold=0.2):
            return value if abs(value) >= threshold else 0.0

        if event.type == pygame.JOYAXISMOTION:
            filtered_value = apply_dead_zone(event.value, threshold=0.2)
            if filtered_value != 0.0:
                print(f"Axis {event.axis} moved, value: {filtered_value}")

pygame.quit()
