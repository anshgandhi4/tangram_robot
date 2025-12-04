import numpy as np

class Piece:
    def __init__(self, coords, color):
        self.coords = coords
        self.color = color
        self.shape = self.classify_shape()
        self.area = self.calculate_area()
        self.pose = None

    def __str__(self):
        return f'Piece(coords={self.coords}, color={self.color}, shape={self.shape}, area={self.area}, pose={self.pose})'

    def __repr__(self):
        return self.__str__()

    def classify_shape(self):
        if len(self.coords) == 3:
            return 'triangle'

        if abs(np.linalg.norm(self.coords[0] - self.coords[1]) - np.linalg.norm(self.coords[1] - self.coords[2])) < 3:
            return 'square'

        return 'parallelogram'

    def calculate_area(self):
        if 'triangle' in self.shape:
            return int(0.5 * np.linalg.det(np.hstack([self.coords, np.ones((3, 1))])))

        if self.shape == 'square':
            return int(np.linalg.norm(self.coords[0] - self.coords[1]) ** 2)

        return int(np.linalg.det(np.hstack([self.coords[:-1], np.ones((3, 1))])))

    def reflect_image(self, image_y):
        self.coords[:, 0] = image_y - self.coords[:, 0]
        self.coords = self.coords[::-1]

    def calculate_pose(self):
        theta = None
        if 'triangle' in self.shape:
            s2 = self.coords[0] - self.coords[1]
            s0 = self.coords[2] - self.coords[1]
            s2_mag = np.linalg.norm(s2)
            s0_mag = np.linalg.norm(s0)

            angle = np.arccos(np.dot(s2, s0) / (s2_mag * s0_mag))
            if abs(angle - 0.5 * np.pi) < 0.05:
                theta = np.arctan2(s2[1], s2[0])
            elif s2_mag > s0_mag:
                theta = np.arctan2(s0[1], s0[0])
            else:
                s1 = self.coords[2] - self.coords[0]
                theta = np.arctan2(s1[1], s1[0])
        elif self.shape == 'square':
            base = self.coords[np.argsort(self.coords[:, 1])][:2]
            base = base[1] - base[0]
            theta = np.arctan2(base[1], base[0])
        else:
            s0 = self.coords[0] - self.coords[1]
            s1 = self.coords[1] - self.coords[2]
            s0_mag = np.linalg.norm(s0)
            s1_mag = np.linalg.norm(s1)

            if s0_mag > s1_mag:
                theta = np.arctan2(s0[1], s0[0])
            else:
                theta = np.arctan2(s1[1], s1[0])

            if theta > 0.5 * np.pi:
                theta -= np.pi
            elif theta < -0.5 * np.pi:
                theta += np.pi

        return np.append(self.coords.mean(axis=0), theta)

class Tangram:
    def __init__(self, pieces=None, prompt=None):
        self.pieces = pieces if pieces is not None else []
        self.prompt = prompt if prompt is not None else ''

    def add_piece(self, piece):
        self.pieces.append(piece)

    def paralellogram_wrong(self):
        for piece in self.pieces:
            if piece.shape == 'parallelogram':
                s2 = piece.coords[0] - piece.coords[1]
                s0 = piece.coords[2] - piece.coords[1]
                s2_mag = np.linalg.norm(s2)
                s0_mag = np.linalg.norm(s0)

                angle = np.arccos(np.dot(s2, s0) / (s2_mag * s0_mag))
                return (s2_mag > s0_mag) == (angle < 0.5 * np.pi)

    def process(self, image_y):
        sizes = ['small', 'small', 'medium', 'large', 'large']
        triangles = [(piece, i) for i, piece in enumerate(self.pieces) if 'triangle' in piece.shape]
        sorted_indices = np.argsort([triangle.area for triangle, _ in triangles])
        for idx, i in enumerate(sorted_indices):
            self.pieces[triangles[i][1]].shape = f'{sizes[idx]} triangle'

        flip = self.paralellogram_wrong()
        if flip:
            for piece in self.pieces:
                piece.reflect_image(image_y)

        for piece in self.pieces:
            piece.pose = piece.calculate_pose()

        return flip
