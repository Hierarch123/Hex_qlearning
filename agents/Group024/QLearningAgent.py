import socket
import random
import time
import pickle

"""
主要需要完善的部分
1.奖励逻辑
2.然后目前移动策略使用的ε-贪婪策略，可以换成根据Q表数据充足程度来选择随机还是最优移动
"""


class QLearningAgent:
    def __init__(self, board_size=11, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.host = "127.0.0.1"
        self.port = 1234
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

        self.board_size = board_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.board = [[0] * board_size for _ in range(board_size)]
        self.colour = ""
        self.turn_count = 0

        self.prev_state = None
        self.prev_action = None

    # 主循环
    def run(self):
        while True:
            start_time = time.time()
            data = self.socket.recv(1024)
            end_time = time.time()
            print(f"Data received in {end_time - start_time} seconds")

            if not data:
                break
            if self.interpret_data(data.decode()):
                break

    # 解释数据
    def interpret_data(self, message):

        commands = message.strip().split("\n")
        for command in commands:
            parts = command.split(";")
            if parts[0] == "START":
                self.colour = parts[2]
                self.init_board()
            elif parts[0] == "CHANGE":
                if parts[3] == "END":
                    # 游戏结束，计算奖励并更新 Q-表
                    reward = self.calculate_reward(None, self.board, True)
                    self.update_q_table(self.prev_state, self.prev_action, reward, self.board)
                    return True
                if parts[1] == "SWAP":
                    self.colour = self.opp_colour()
                    self.swap_board()
                else:
                    self.update_board(parts[1])
                if parts[3] == self.colour:
                    # 计算上一步的奖励并更新 Q-表
                    reward = self.calculate_reward(self.prev_action, self.board, False)
                    self.update_q_table(self.prev_state, self.prev_action, reward, self.board)

                    # 准备下一步行动
                    self.make_move()
            elif parts[0] == "END":
                return True
        return False

    # 交换棋盘颜色
    def swap_board(self):
        self.board = [[-cell for cell in row] for row in self.board]

    # 获取对方颜色
    def opp_colour(self):
        return "B" if self.colour == "R" else "R"

    # 初始化棋盘
    # 游戏开始或颜色交换后重置棋盘状态
    def init_board(self):
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        if self.colour == "R":
            self.make_move()

    # 根据对手的最新移动更新棋盘状态。
    def update_board(self, action):
        if action != "SWAP":
            x, y = map(int, action.split(","))
            self.board[x][y] = 1 if self.colour == "R" else -1

    # 执行移动
    def make_move(self):
        self.prev_state = self.state_to_key(self.board)

        # ε-贪婪策略
        if random.random() < self.epsilon:
            move = self.random_move()
        else:
            move = self.best_move()

        self.prev_action = move
        self.socket.sendall(f"{move[0]},{move[1]}\n".encode())
        self.board[move[0]][move[1]] = 1 if self.colour == "R" else -1

    def random_move(self):
        empty_cells = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if self.board[x][y] == 0]
        return random.choice(empty_cells)

    # 基于Q表选择最佳移动
    def best_move(self):
        possible_actions = self.get_possible_actions(self.board)
        if not possible_actions:
            return None

        state_key = self.state_to_key(self.board)
        best_action = None
        best_q_value = float('-inf')

        for action in possible_actions:
            action_key = self.action_to_key(action)
            q_value = self.q_table.get((state_key, action_key), 0)

            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action if best_action is not None else self.random_move()

    def calculate_reward(self, action, new_state, end_game):

        reward = 0

        # 检查游戏是否结束
        if end_game:
            if self.has_won(new_state):
                reward += 100  # 获胜奖励
            else:
                reward -= 100  # 失败惩罚

        return reward

    def has_won(self, state):

        pass

    # 更新Q表
    def update_q_table(self, state, action, reward, new_state):

        state_key = self.state_to_key(state)
        new_state_key = self.state_to_key(new_state)
        action_key = self.action_to_key(action)

        # 获取当前状态和行动对应的Q值
        current_q = self.q_table.get((state_key, action_key), 0)

        # 计算新状态的最大Q值
        max_new_q = max(self.q_table.get((new_state_key, a), 0) for a in self.get_possible_actions(new_state))

        # 更新Q值
        self.q_table[(state_key, action_key)] = current_q + self.alpha * (reward + self.gamma * max_new_q - current_q)

    # 将棋盘状态转换为一个字符串键。
    def state_to_key(self, state):
        return ''.join(str(cell) for row in state for cell in row)

    # 将行动转换为 Q-表中的键。
    def action_to_key(self, action):
        return action

    def get_possible_actions(self, state):

        # 返回当前状态下所有可能的合法行动。
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if state[i][j] == 0]

    def save_q_table(self, filename):

        with open(filename, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_q_table(self, filename):
        # 从文件加载 Q-表
        try:
            with open(filename, 'rb') as file:
                self.q_table = pickle.load(file)
        except FileNotFoundError:
            self.q_table = {}


if __name__ == "__main__":
    q_table_filename = 'q_table.pkl'
    agent = QLearningAgent()
    agent.load_q_table(q_table_filename)  # 加载 Q-表
    agent.run()
    agent.save_q_table(q_table_filename)  # 保存 Q-表
