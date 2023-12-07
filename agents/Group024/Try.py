import socket
import random
import time

"""
目前没有完成的部分：
1.best_move 方法：根据Q-表和当前棋盘状态，选择预期奖励最大的行动。
2.update_q_table 方法：根据代理的行动、接收到的奖励和新的游戏状态，更新 Q-表中的值。
3.定义奖励机制
4.探索策略: 目前使用随机探索策略，没有考虑 Q-表
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
        self.q_table = {}  # 简化的Q表
        self.board = [[0] * board_size for _ in range(board_size)]
        self.colour = ""
        self.turn_count = 0

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
        """
        解释从游戏引擎接收的数据，并根据数据做出响应。
        """
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
    # 暂时使用随机移动代替
    def make_move(self):
        # 保存当前状态
        self.prev_state = self.state_to_key(self.board)

        # 选择并执行行动
        move = self.random_move()  # 或 best_move，取决于你的实现
        self.prev_action = move
        self.socket.sendall(f"{move[0]},{move[1]}\n".encode())

        # 更新棋盘
        self.board[move[0]][move[1]] = 1 if self.colour == "R" else -1

    def random_move(self):
        empty_cells = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if self.board[x][y] == 0]
        return random.choice(empty_cells)

    # 实现最佳移动的逻辑，目前使用随机移动代替
    def best_move(self):

        return self.random_move()

    def calculate_reward(self, action, new_state, end_game):
        """
        根据行动、新状态和游戏结束情况计算奖励。

        :param action: 行动，格式为 (x, y)
        :param new_state: 新状态，即更新后的棋盘
        :param end_game: 游戏是否结束
        :return: 计算出的奖励值
        """
        reward = 0

        # 检查是否接近胜利或阻止了对手
        # 这需要一些复杂的逻辑来检测连线的情况
        # reward += ...

        # 检查游戏是否结束
        if end_game:
            if self.has_won(new_state):
                reward += 100  # 获胜奖励
            else:
                reward -= 100  # 失败惩罚

        # 检查是否进行了无效或非法行动
        # 例如，如果action在棋盘上的位置已被占用
        # if self.board[action[0]][action[1]] != 0:
        #     reward -= 50

        return reward

    def has_won(self, state):
        """
        检查当前玩家是否在给定状态下赢得了游戏。
        这需要实现具体的胜利条件检查。
        """
        # 实现胜利条件的检查逻辑
        # return True/False
        pass

    # 更新Q表
    def update_q_table(self, state, action, reward, new_state):
        """
        更新 Q-表的方法。

        :param state: 当前状态
        :param action: 当前执行的行动
        :param reward: 接收到的奖励
        :param new_state: 新的状态
        """
        state_key = self.state_to_key(state)
        new_state_key = self.state_to_key(new_state)
        action_key = self.action_to_key(action)

        # 获取当前状态和行动对应的 Q-值
        current_q = self.q_table.get((state_key, action_key), 0)

        # 计算新状态的最大 Q-值
        max_new_q = max(self.q_table.get((new_state_key, a), 0) for a in self.get_possible_actions(new_state))

        # 更新 Q-值
        self.q_table[(state_key, action_key)] = current_q + self.alpha * (reward + self.gamma * max_new_q - current_q)

    def state_to_key(self, state):
        """
        将棋盘状态转换为一个字符串键。
        """
        return ''.join(str(cell) for row in state for cell in row)

    def action_to_key(self, action):
        """
        将行动转换为 Q-表中的键。行动通常是棋盘上的位置，可以直接使用。
        """
        return action

    def get_possible_actions(self, state):
        """
        返回当前状态下所有可能的合法行动。
        """
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if state[i][j] == 0]


if __name__ == "__main__":
    agent = QLearningAgent()
    agent.run()
