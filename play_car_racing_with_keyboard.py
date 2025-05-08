import gym
import pyglet.window

is_pressed_left  = False # control left
is_pressed_right = False # control right
is_pressed_space = False # control gas
is_pressed_shift = False # control break
is_pressed_esc   = False # exit the game
steering_wheel = 0 # init to 0
gas            = 0 # init to 0
break_system   = 0 # init to 0

def key_press(key, mod):
    print(f"key_press: {key}, mod: {mod}")  # 디버깅용
    global is_pressed_left
    global is_pressed_right
    global is_pressed_space
    global is_pressed_shift
    global is_pressed_esc

    if key == pyglet.window.key.LEFT:
        is_pressed_left = True
    if key == pyglet.window.key.RIGHT:
        is_pressed_right = True
    if key == pyglet.window.key.SPACE:
        is_pressed_space = True
    if key == pyglet.window.key.LSHIFT or key == pyglet.window.key.RSHIFT:
        is_pressed_shift = True
    if key == pyglet.window.key.ESCAPE:
        is_pressed_esc = True

def key_release(key, mod):
    print(f"key_release: {key}, mod: {mod}")  # 디버깅용
    global is_pressed_left
    global is_pressed_right
    global is_pressed_space
    global is_pressed_shift

    if key == pyglet.window.key.LEFT:
        is_pressed_left = False
    if key == pyglet.window.key.RIGHT:
        is_pressed_right = False
    if key == pyglet.window.key.SPACE:
        is_pressed_space = False
    if key == pyglet.window.key.LSHIFT or key == pyglet.window.key.RSHIFT:
        is_pressed_shift = False

def update_action():
    global steering_wheel
    global gas
    global break_system

    if is_pressed_left ^ is_pressed_right:
        if is_pressed_left:
            if steering_wheel > -1:
                steering_wheel -= 0.1
            else:
                steering_wheel = -1
        if is_pressed_right:
            if steering_wheel < 1:
                steering_wheel += 0.1
            else:
                steering_wheel = 1
    else:
        if abs(steering_wheel - 0) < 0.1:
            steering_wheel = 0
        elif steering_wheel > 0:
            steering_wheel -= 0.1
        elif steering_wheel < 0:
            steering_wheel += 0.1
    if is_pressed_space:
        if gas < 1:
            gas += 0.1
        else:
            gas = 1
    else:
        if gas > 0:
            gas -= 0.1
        else:
            gas = 0
    if is_pressed_shift:
        if break_system < 1:
            break_system += 0.1
        else:
            break_system = 1
    else:
        if break_system > 0:
            break_system -= 0.1
        else:
            break_system = 0

if __name__ == '__main__':
    env = gym.make('CarRacing-v2', render_mode='human')
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    env.render()  # viewer 생성 시도

    # viewer가 생성될 때까지 반복적으로 render 호출
    while not hasattr(env.unwrapped, 'viewer') or env.unwrapped.viewer is None or not hasattr(env.unwrapped.viewer, 'window'):
        env.render()

    env.unwrapped.viewer.window.push_handlers(
        on_key_press=key_press,
        on_key_release=key_release
    )
    env.unwrapped.viewer.zoom = 2.0  # 차를 더 크게 보이게

    counter = 0
    total_reward = 0
    while not is_pressed_esc:
        env.render()
        update_action()
        action = [steering_wheel, gas, break_system]
        step_result = env.step(action)
        if len(step_result) == 5:
            state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            state, reward, done, info = step_result
        counter += 1
        total_reward += reward
        print('Action:[{:+.1f}, {:+.1f}, {:+.1f}] Reward: {:.3f}'.format(action[0], action[1], action[2], reward))
        if done:
            print("Restart game after {} timesteps. Total Reward: {}".format(counter, total_reward))
            counter = 0
            total_reward = 0
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            continue

    env.close()
