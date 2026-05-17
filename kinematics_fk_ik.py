"""
kinematics_fk_ik.py
Прямая (FK) и обратная (IK) кинематика для 2-звенного робота.
Демонстрация загрузки URDF, вычисления FK и итеративного IK.
"""

import torch
import pytorch_kinematics as pk
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. ЗАГРУЗКА МОДЕЛИ РОБОТА ИЗ URDF
# ============================================================
print("=" * 60)
print("1. ЗАГРУЗКА МОДЕЛИ РОБОТА")
print("=" * 60)

# Загружаем кинематическую цепочку из URDF-файла
# Первый аргумент: содержимое URDF-файла
# Второй аргумент: имя конечного звена (end effector)
urdf_path = "robots/my_robot.urdf"
chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), "tool0")

# Информация о роботе
print(f"Имена сочленений: {chain.get_joint_parameter_names()}")
print(f"Количество сочленений (DOF): {chain.n_joints}")
print(f"Ограничения углов (рад):\n{chain.get_joint_limits()}")

# ============================================================
# 2. ПРЯМАЯ КИНЕМАТИКА (FK)
# ============================================================
print("\n" + "=" * 60)
print("2. ПРЯМАЯ КИНЕМАТИКА (FK)")
print("=" * 60)

# Задаём углы сочленений в радианах: 30° и 45°
theta1 = 30 * np.pi / 180  # 0.5236 рад
theta2 = 45 * np.pi / 180  # 0.7854 рад
joint_angles = torch.tensor([theta1, theta2])

print(f"Входные углы: θ1 = {theta1:.3f} рад ({theta1*180/np.pi:.0f}°), "
      f"θ2 = {theta2:.3f} рад ({theta2*180/np.pi:.0f}°)")

# Вычисляем FK
transform = chain.forward_kinematics(joint_angles)
matrix = transform.get_matrix()
position = matrix[:, :3, 3]  # извлекаем x, y, z

print(f"Положение схвата (tool0):")
print(f"  x = {position[0,0]:.4f} м")
print(f"  y = {position[0,1]:.4f} м")
print(f"  z = {position[0,2]:.4f} м")

# ============================================================
# 3. ОБРАТНАЯ КИНЕМАТИКА (IK) — поиск углов по целевой позиции
# ============================================================
print("\n" + "=" * 60)
print("3. ОБРАТНАЯ КИНЕМАТИКА (IK)")
print("=" * 60)

# Целевая позиция схвата
target_x = 1.3
target_y = 0.9
target_pos = torch.tensor([target_x, target_y, 0.0])

print(f"Целевая позиция: x = {target_x}, y = {target_y}")

# Начальная догадка (углы)
q = torch.tensor([0.0, 0.0], requires_grad=True)

# Параметры IK
max_iterations = 50
learning_rate = 0.5
tolerance = 0.001  # 1 мм точность

print(f"\nНачинаем итерации (макс {max_iterations}, точность {tolerance} м)...")
print("-" * 50)

history_error = []

for i in range(max_iterations):
    # 1. Прямая кинематика: текущее положение
    transform = chain.forward_kinematics(q)
    current_pos = transform.get_matrix()[:, :3, 3].squeeze()
    
    # 2. Ошибка между текущим и целевым положением
    error = target_pos - current_pos
    error_norm = torch.norm(error).item()
    history_error.append(error_norm)
    
    # 3. Проверка сходимости
    if error_norm < tolerance:
        print(f"\n✅ Достигнута точность на итерации {i+1}")
        break
    
    # 4. Вычисляем Якобиан и псевдообратную матрицу (только позиционная часть)
    J = chain.jacobian(q).squeeze(0)  # (6, n)
    J_pos = J[:3, :]                   # берём только строки для x, y, z
    J_plus = torch.linalg.pinv(J_pos)  # (n, 3)
    
    # 5. Изменение углов
    delta_q = J_plus @ error
    q = q + learning_rate * delta_q
    
    if i % 10 == 0:
        print(f"Итерация {i+1:3d}: ошибка = {error_norm:.6f} м")

print(f"\nРезультат IK:")
print(f"  Найденные углы: θ1 = {q[0].item():.4f} рад ({q[0].item()*180/np.pi:.1f}°), "
      f"θ2 = {q[1].item():.4f} рад ({q[1].item()*180/np.pi:.1f}°)")

# Проверка: FK по найденным углам
final_transform = chain.forward_kinematics(q)
final_pos = final_transform.get_matrix()[:, :3, 3].squeeze()
print(f"  Достигнутая позиция: x = {final_pos[0].item():.4f}, y = {final_pos[1].item():.4f}")
print(f"  Конечная ошибка: {torch.norm(target_pos - final_pos).item():.6f} м")

# ============================================================
# 4. ВИЗУАЛИЗАЦИЯ СХОДИМОСТИ
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(history_error, 'b-', linewidth=2)
plt.yscale('log')
plt.xlabel('Итерация')
plt.ylabel('Ошибка (м)')
plt.title('Сходимость обратной кинематики (логарифмическая шкала)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ik_convergence.png')
print(f"\nГрафик сходимости сохранён в 'ik_convergence.png'")