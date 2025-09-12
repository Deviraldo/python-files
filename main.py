"""
smart_traffic_demo.py

Requirements (install if needed):
pip install opencv-python tensorflow numpy pygame requests folium streamlit plotly pandas
(You can omit ones you don't use - this demo uses: numpy, pygame, tensorflow(optional for RL), pandas)

Run:
python smart_traffic_demo.py
"""

import math
import random
import time
from collections import deque

import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

# If TensorFlow is heavy for you or unavailable, you can stub out the RL controller.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None
    layers = None
    models = None
    print("TensorFlow not available — RL model will be a random policy fallback.")


# ---------------------------
# RoadObjectDetector (simulated)
# ---------------------------
class RoadObjectDetector:
    """Simulated detector returning objects in world coordinates (x, y, z)."""

    def _init_(self):
        self.classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'traffic_light']
        # We keep it deterministic-ish for demo reproducibility
        random.seed(0)

    def simulate_scene(self, sim_manager):
        """
        Return a list of detection dicts based on the SimulationManager internal state.
        Each detection contains:
            - 'class'
            - 'center_world': (x, y, z)
            - 'speed' (km/h) optional
            - 'state' for traffic_light
        """
        detections = []

        # vehicles
        for v in sim_manager.vehicles:
            det = {
                'class': v['type'],
                'center_world': (v['x'], v['y'], 0),
                'speed': v['speed_kmph'],
                'id': v['id'],
                'bbox': [0, 0, 0, 0]
            }
            detections.append(det)

        # pedestrians
        for p in sim_manager.pedestrians:
            det = {
                'class': 'person',
                'center_world': (p['x'], p['y'], 0),
                'speed': p['speed_kmph'],
                'id': p['id'],
                'bbox': [0, 0, 0, 0]
            }
            detections.append(det)

        # traffic lights (only one intersection in demo)
        tl = sim_manager.traffic_light
        detections.append({
            'class': 'traffic_light',
            'center_world': (tl['world_x'], tl['world_y'], 2),
            'state': tl['state'],
            'id': 'TL_1'
        })

        return detections


# ---------------------------
# ReinforcementLearningTrafficController (slimmed)
# ---------------------------
class ReinforcementLearningTrafficController:
    """Simplified DQN controller — if TF missing, falls back to random policy."""

    def _init_(self, num_intersections=1):
        self.num_intersections = num_intersections
        self.state_size = 16
        self.action_size = 4
        self.memory = deque(maxlen=5000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95

        # Build networks if TF available
        if tf is not None and layers is not None and models is not None:
            self.q_network = self.build_model()
            self.target_network = self.build_model()
            self.update_target_network()
            self.use_tf = True
        else:
            self.q_network = None
            self.target_network = None
            self.use_tf = False

    def build_model(self):
        model = models.Sequential([
            layers.Dense(128, input_dim=self.state_size, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def update_target_network(self):
        if self.use_tf:
            self.target_network.set_weights(self.q_network.get_weights())

    def get_state(self, traffic_data):
        """traffic_data: dict — produce vector of length state_size"""
        state = np.zeros(self.state_size, dtype=np.float32)
        state[0] = traffic_data.get('vehicle_count_ns', 0) / 20.0
        state[1] = traffic_data.get('vehicle_count_ew', 0) / 20.0
        state[2] = traffic_data.get('pedestrian_count', 0) / 10.0
        state[3] = traffic_data.get('avg_speed_ns', 0) / 60.0
        state[4] = traffic_data.get('avg_speed_ew', 0) / 60.0
        state[5] = traffic_data.get('queue_length_ns', 0) / 15.0
        state[6] = traffic_data.get('queue_length_ew', 0) / 15.0
        state[7] = traffic_data.get('wait_time_ns', 0) / 120.0
        state[8] = traffic_data.get('wait_time_ew', 0) / 120.0
        state[9] = traffic_data.get('current_phase', 0) / 3.0
        state[10] = traffic_data.get('time_in_phase', 0) / 60.0
        state[11] = float(bool(traffic_data.get('emergency_vehicles', 0)))
        state[12] = traffic_data.get('weather_factor', 1.0)
        state[13] = traffic_data.get('time_of_day', 12) / 24.0
        state[14] = traffic_data.get('day_of_week', 0) / 7.0
        state[15] = traffic_data.get('historical_flow', 0) / 100.0
        return state

    def act(self, state):
        # epsilon-greedy
        if np.random.rand() <= self.epsilon or not self.use_tf:
            return random.randrange(self.action_size)
        q = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return int(np.argmax(q[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self, batch_size=32):
        if not self.use_tf:
            # no training without TF in this demo
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        q_curr = self.q_network.predict(states, verbose=0)
        q_next = self.target_network.predict(next_states, verbose=0)
        for i in range(batch_size):
            if dones[i]:
                q_curr[i][actions[i]] = rewards[i]
            else:
                q_curr[i][actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
        self.q_network.fit(states, q_curr, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ---------------------------
# SimulationManager
# ---------------------------
class SimulationManager:
    """Simple traffic simulation for one intersection"""

    def _init_(self):
        self.vehicles = []  # list of dicts
        self.pedestrians = []
        self.next_vehicle_id = 1
        self.next_ped_id = 1
        self.spawn_timer = 0.0
        # Intersection coordinates in world space (center=0,0)
        # Traffic light placed near north-west corner of intersection
        self.traffic_light = {
            'state': 'green',  # 'green', 'yellow', 'red'
            'current_phase': 0,  # 0: NS green, 1: NS yellow, 2: EW green, 3: EW yellow
            'phase_time': 0.0,
            'phase_length': {0: 8.0, 1: 2.0, 2: 8.0, 3: 2.0},
            'world_x': 0,
            'world_y': -80  # top of screen in world coords
        }
        self.time = 0.0
        self.metrics = {
            'total_wait_time': 0.0,
            'throughput': 0,
            'emissions': 0.0,
            'safety_violations': 0
        }
        self.vehicle_spawn_rate = 1.0  # vehicles per second average
        self.ped_spawn_rate = 0.1
        self.random = random.Random(1)

    def spawn_vehicle(self):
        # spawn at random edge and move towards center
        edges = ['north', 'south', 'east', 'west']
        edge = self.random.choice(edges)
        v_type = self.random.choice(['car', 'car', 'truck', 'motorcycle'])
        speed_kmph = {'car': 40, 'truck': 30, 'motorcycle': 50}.get(v_type, 35)
        # Convert km/h to world units per second (we treat 1 world unit = 1 pixel, and assume 1s sim step)
        # This is arbitrary scaling for demo
        vx, vy = 0.0, 0.0
        if edge == 'north':
            x, y = self.random.uniform(-250, 250), -500
            vx, vy = 0, 1
        elif edge == 'south':
            x, y = self.random.uniform(-250, 250), 500
            vx, vy = 0, -1
        elif edge == 'east':
            x, y = 500, self.random.uniform(-250, 250)
            vx, vy = -1, 0
        else:  # west
            x, y = -500, self.random.uniform(-250, 250)
            vx, vy = 1, 0

        v = {
            'id': f"V{self.next_vehicle_id}",
            'type': v_type,
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'speed_kmph': speed_kmph,
            'waiting_since': None,
            'passed_center': False
        }
        self.next_vehicle_id += 1
        self.vehicles.append(v)

    def spawn_pedestrian(self):
        # spawn pedestrian near the crossing, crossing across
        side = self.random.choice(['ns', 'ew'])
        if side == 'ns':
            x = self.random.uniform(-60, 60)
            y = -120
            vx, vy = 0, 0.5
        else:
            x = -120
            y = self.random.uniform(-60, 60)
            vx, vy = 0.5, 0
        p = {
            'id': f"P{self.next_ped_id}",
            'x': x, 'y': y, 'vx': vx, 'vy': vy,
            'speed_kmph': 5
        }
        self.next_ped_id += 1
        self.pedestrians.append(p)

    def update(self, dt, controller_action=None):
        """
        dt: seconds elapsed
        controller_action: integer 0..3 to set traffic light phase
        """
        self.time += dt
        # Possibly spawn vehicles/peds
        self.spawn_timer += dt
        # spawn vehicles based on poisson-ish rate
        while self.spawn_timer > 1.0 / max(1e-6, self.vehicle_spawn_rate):
            if random.random() < 0.7:
                self.spawn_vehicle()
            if random.random() < self.ped_spawn_rate:
                self.spawn_pedestrian()
            self.spawn_timer -= 1.0 / max(1e-6, self.vehicle_spawn_rate)

        # Apply controller action if present (0..3)
        if controller_action is not None:
            # map action to phase directly
            ph = int(controller_action) % 4
            if ph != self.traffic_light['current_phase']:
                self.traffic_light['current_phase'] = ph
                self.traffic_light['phase_time'] = 0.0
                # map phase to state
                if ph == 0:
                    self.traffic_light['state'] = 'green'
                elif ph == 1:
                    self.traffic_light['state'] = 'yellow'
                elif ph == 2:
                    self.traffic_light['state'] = 'green'
                elif ph == 3:
                    self.traffic_light['state'] = 'yellow'

        # advance phase time
        self.traffic_light['phase_time'] += dt
        cur_phase = self.traffic_light['current_phase']
        phase_len = self.traffic_light['phase_length'].get(cur_phase, 6.0)
        if self.traffic_light['phase_time'] >= phase_len:
            # auto-advance
            self.traffic_light['current_phase'] = (cur_phase + 1) % 4
            self.traffic_light['phase_time'] = 0.0
            # set green/yellow appropriately
            if self.traffic_light['current_phase'] in (0, 2):
                self.traffic_light['state'] = 'green'
            else:
                self.traffic_light['state'] = 'yellow'

        # move vehicles
        removed = []
        for v in self.vehicles:
            # Determine if should stop at intersection due to red light
            stop = False
            # if vehicle is approaching center (within 120 units) and light for their direction is red, they stop
            approaching_ns = abs(v['x']) < 80 and ((v['vy'] < 0 and v['y'] < 0) or (v['vy'] > 0 and v['y'] > 0))
            approaching_ew = abs(v['y']) < 80 and ((v['vx'] < 0 and v['x'] < 0) or (v['vx'] > 0 and v['x'] > 0))

            # Decide which directions are green given phase:
            phase = self.traffic_light['current_phase']
            ns_green = (phase == 0)
            ew_green = (phase == 2)

            # If approaching center and facing a red, stop
            if approaching_ns and not ns_green:
                # vehicles in north/south direction should stop when not NS green
                # apply stopping if within small distance to center
                dist_to_center = abs(v['y'])
                if dist_to_center < 160:
                    stop = True

            if approaching_ew and not ew_green:
                dist_to_center = abs(v['x'])
                if dist_to_center < 160:
                    stop = True

            # apply motion
            if stop:
                # mark waiting start
                if v['waiting_since'] is None:
                    v['waiting_since'] = self.time
                # accumulate wait time metric (per-second)
                self.metrics['total_wait_time'] += dt
            else:
                if v['waiting_since'] is not None:
                    # vehicle resumed, clear waiting_since
                    v['waiting_since'] = None
                # speed -> world units per second (approx)
                world_speed = v['speed_kmph'] * (1000.0 / 3600.0) * 0.05  # scaled down
                # move
                v['x'] += v['vx'] * world_speed * dt * 30.0
                v['y'] += v['vy'] * world_speed * dt * 30.0

            # mark passed center and increment throughput when they cross central line
            if not v['passed_center'] and abs(v['x']) < 10 and abs(v['y']) < 10:
                v['passed_center'] = True
                self.metrics['throughput'] += 1

            # remove vehicles that are far beyond the map
            if abs(v['x']) > 1000 or abs(v['y']) > 1000:
                removed.append(v)

        for v in removed:
            try:
                self.vehicles.remove(v)
            except ValueError:
                pass

        # pedestrians move simply and are removed when far
        ped_removed = []
        for p in self.pedestrians:
            p['x'] += p['vx'] * 40 * dt
            p['y'] += p['vy'] * 40 * dt
            if abs(p['x']) > 1000 or abs(p['y']) > 1000:
                ped_removed.append(p)
        for p in ped_removed:
            try:
                self.pedestrians.remove(p)
            except ValueError:
                pass

        # emissions simple incremental proportional to number of vehicles
        self.metrics['emissions'] += len(self.vehicles) * 0.001 * dt

    def get_traffic_data_for_controller(self):
        """Build a traffic_data dict the RL controller expects"""
        # Very simple estimations for demo
        ns_count = sum(1 for v in self.vehicles if abs(v['x']) < abs(v['y']))  # rough heuristic
        ew_count = max(0, len(self.vehicles) - ns_count)
        avg_ns_speed = np.mean([v['speed_kmph'] for v in self.vehicles if abs(v['x']) < abs(v['y'])] or [0])
        avg_ew_speed = np.mean([v['speed_kmph'] for v in self.vehicles if abs(v['x']) >= abs(v['y'])] or [0])
        queue_ns = max(0, int(ns_count * 0.3))
        queue_ew = max(0, int(ew_count * 0.3))
        wait_ns = queue_ns * 5.0
        wait_ew = queue_ew * 5.0
        td = {
            'vehicle_count_ns': ns_count,
            'vehicle_count_ew': ew_count,
            'pedestrian_count': len(self.pedestrians),
            'avg_speed_ns': avg_ns_speed,
            'avg_speed_ew': avg_ew_speed,
            'queue_length_ns': queue_ns,
            'queue_length_ew': queue_ew,
            'wait_time_ns': wait_ns,
            'wait_time_ew': wait_ew,
            'current_phase': self.traffic_light['current_phase'],
            'time_in_phase': self.traffic_light['phase_time'],
            'emergency_vehicles': 0,
            'weather_factor': 1.0,
            'time_of_day': 12,
            'day_of_week': 2,
            'historical_flow': max(len(self.vehicles), 0)
        }
        return td


# ---------------------------
# Traffic3DVisualizer
# ---------------------------
class Traffic3DVisualizer:
    """3D-ish visualization using Pygame + simple projection"""

    def _init_(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Smart Traffic Management - 3D Demo")
        self.clock = pygame.time.Clock()
        self.running = True

        # Colors
        self.colors = {
            'road': (60, 60, 60),
            'lane': (255, 255, 0),
            'sky': (135, 206, 250),
            'grass': (34, 139, 34),
            'car': (200, 0, 0),
            'truck': (20, 60, 200),
            'motorcycle': (255, 165, 0),
            'pedestrian': (255, 20, 147),
            'traffic_light_red': (255, 0, 0),
            'traffic_light_yellow': (255, 255, 0),
            'traffic_light_green': (0, 200, 0),
            'building': (130, 130, 130)
        }

        # Camera and projection parameters
        self.camera_pos = [0, 0, 800]  # looking down from z=800 in world coords
        self.camera_height = 800
        self.scale = 0.6  # scaling world coords to screen
        # world center maps to screen center
        self.screen_center = (self.width // 2, self.height // 2)
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 14)

    def world_to_screen(self, x, y, z=0):
        """Convert world coordinates (x,y,z) to screen coordinates (sx, sy). Simplified orthographic-ish projection."""
        # Translate relative to camera but we use top-down view: x -> x, y -> y
        sx = int(self.screen_center[0] + x * self.scale)
        sy = int(self.screen_center[1] + y * self.scale)
        return sx, sy

    def draw_road_and_buildings(self):
        # sky/ground
        self.screen.fill(self.colors['sky'])
        # draw grass border
        pygame.draw.rect(self.screen, self.colors['grass'], (0, 0, self.width, self.height))
        # draw road cross (centered)
        rs = self.world_to_screen(-400, -100)
        re = self.world_to_screen(400, 100)
        # north-south road (rectangle)
        top_left = self.world_to_screen(-50, -400)
        bottom_right = self.world_to_screen(50, 400)
        pygame.draw.rect(self.screen, self.colors['road'],
                         (top_left[0], top_left[1], (bottom_right[0] - top_left[0]), (bottom_right[1] - top_left[1])))

        left = self.world_to_screen(-400, -50)
        right = self.world_to_screen(400, 50)
        pygame.draw.rect(self.screen, self.colors['road'],
                         (left[0], left[1], (right[0] - left[0]), (right[1] - left[1])))

        # lane dividers (simple lines)
        for i in range(-200, 201, 20):
            sx1, sy1 = self.world_to_screen(i, -5)
            sx2, sy2 = self.world_to_screen(i + 10, -5)
            pygame.draw.line(self.screen, self.colors['lane'], (sx1, sy1), (sx2, sy2), 2)
            sx1, sy1 = self.world_to_screen(i, 5)
            sx2, sy2 = self.world_to_screen(i + 10, 5)
            pygame.draw.line(self.screen, self.colors['lane'], (sx1, sy1), (sx2, sy2), 2)

        # buildings at corners
        buildings = [
            (200, 200, 80, 80), (-200, 200, 100, 60), (200, -200, 60, 110), (-200, -200, 90, 90)
        ]
        for bx, by, bw, bh in buildings:
            sx, sy = self.world_to_screen(bx - bw // 2, by - bh // 2)
            sw, sh = int(bw * self.scale), int(bh * self.scale)
            pygame.draw.rect(self.screen, self.colors['building'], (sx, sy, sw, sh))

    def draw_vehicle(self, v):
        # draw box depending on type using world coordinates
        sx, sy = self.world_to_screen(v['x'], v['y'])
        if v['type'] == 'truck':
            w, h = 30, 14
            color = self.colors['truck']
        elif v['type'] == 'motorcycle':
            w, h = 12, 6
            color = self.colors['motorcycle']
        else:
            w, h = 20, 10
            color = self.colors['car']
        pygame.draw.rect(self.screen, color, (sx - w // 2, sy - h // 2, w, h))
        # id text
        id_surf = self.font.render(v['id'], True, (0, 0, 0))
        self.screen.blit(id_surf, (sx - w // 2, sy - h // 2 - 12))

    def draw_pedestrian(self, p):
        sx, sy = self.world_to_screen(p['x'], p['y'])
        pygame.draw.circle(self.screen, self.colors['pedestrian'], (sx, sy), 4)
        id_surf = self.font.render(p['id'], True, (0, 0, 0))
        self.screen.blit(id_surf, (sx + 6, sy - 6))

    def draw_traffic_light(self, tl):
        sx, sy = self.world_to_screen(tl['world_x'], tl['world_y'])
        # small box
        w, h = 12, 28
        pygame.draw.rect(self.screen, (30, 30, 30), (sx - w // 2, sy - h // 2, w, h))
        # lights positions
        if tl['state'] == 'red':
            pygame.draw.circle(self.screen, self.colors['traffic_light_red'], (sx, sy - 8), 4)
            pygame.draw.circle(self.screen, (80, 80, 80), (sx, sy), 4)
            pygame.draw.circle(self.screen, (80, 80, 80), (sx, sy + 8), 4)
        elif tl['state'] == 'yellow':
            pygame.draw.circle(self.screen, (80, 80, 80), (sx, sy - 8), 4)
            pygame.draw.circle(self.screen, self.colors['traffic_light_yellow'], (sx, sy), 4)
            pygame.draw.circle(self.screen, (80, 80, 80), (sx, sy + 8), 4)
        else:
            pygame.draw.circle(self.screen, (80, 80, 80), (sx, sy - 8), 4)
            pygame.draw.circle(self.screen, (80, 80, 80), (sx, sy), 4)
            pygame.draw.circle(self.screen, self.colors['traffic_light_green'], (sx, sy + 8), 4)

    def render(self, sim_manager):
        # draw background, roads, buildings
        self.draw_road_and_buildings()

        # draw vehicles
        for v in sim_manager.vehicles:
            self.draw_vehicle(v)

        # draw pedestrians
        for p in sim_manager.pedestrians:
            self.draw_pedestrian(p)

        # draw traffic light
        self.draw_traffic_light(sim_manager.traffic_light)

        # draw HUD metrics
        hud_lines = [
            f"Vehicles: {len(sim_manager.vehicles)}",
            f"Pedestrians: {len(sim_manager.pedestrians)}",
            f"Throughput: {sim_manager.metrics['throughput']}",
            f"Total wait: {sim_manager.metrics['total_wait_time']:.1f}s",
            f"Traffic Light Phase: {sim_manager.traffic_light['current_phase']} ({sim_manager.traffic_light['state']})",
        ]
        for i, ln in enumerate(hud_lines):
            surf = self.font.render(ln, True, (0, 0, 0))
            self.screen.blit(surf, (10, 10 + i * 18))

        pygame.display.flip()


# ---------------------------
# Main demo loop wiring everything
# ---------------------------
def main():
    viz = Traffic3DVisualizer(width=900, height=600)
    sim = SimulationManager()
    detector = RoadObjectDetector()
    controller = ReinforcementLearningTrafficController()

    # We'll step at ~30 FPS
    fps = 30
    dt = 1.0 / fps
    last_time = time.time()

    # Seed simulation with a few vehicles
    for _ in range(6):
        sim.spawn_vehicle()

    # main loop
    try:
        while viz.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    viz.running = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    viz.running = False

            # build state for controller
            traffic_data = sim.get_traffic_data_for_controller()
            state = controller.get_state(traffic_data)

            # controller chooses an action every step (could be less frequent)
            action = controller.act(state)

            # update sim with chosen action
            sim.update(dt, controller_action=action)

            # sample detection (not used to drive sim; only for potential future use)
            detections = detector.simulate_scene(sim)

            # optional RL memory update (store transition)
            next_td = sim.get_traffic_data_for_controller()
            next_state = controller.get_state(next_td)

            # compute reward simply: negative of total_wait_time change minus safety penalty
            reward = (traffic_data.get('vehicle_count_ns', 0) + traffic_data.get('vehicle_count_ew', 0)) * 0.0
            # encourage throughput
            reward += (next_td.get('historical_flow', 0) - traffic_data.get('historical_flow', 0)) * 0.5
            reward -= (next_td.get('wait_time_ns', 0) + next_td.get('wait_time_ew', 0)) * 0.01

            done = False
            controller.remember(state, action, reward, next_state, done)
            controller.replay(batch_size=16)

            # visualize
            viz.render(sim)

            # frame limiter
            viz.clock.tick(fps)
    finally:
        pygame.quit()


if _name_ == "_main_":
    main()


    