import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="Custom")
class SpiralPatchify(keras.layers.Layer):
    def __init__(self, patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)

    def _spiral_order(self, n_rows, n_cols):
        # Для чётной сетки берём верхний левый из центральных патчей.
        # Например, для 8x8 старт будет (3, 3) в 0-индексации.
        start_r = (n_rows - 1) // 2
        start_c = (n_cols - 1) // 2

        total = n_rows * n_cols
        visited = [[False for _ in range(n_cols)] for _ in range(n_rows)]
        order = []

        r = start_r
        c = start_c

        if not (0 <= r < n_rows and 0 <= c < n_cols):
            raise ValueError("Start patch is outside grid.")

        visited[r][c] = True
        order.append(r * n_cols + c)

        # вправо, вниз, влево, вверх
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        step_len = 1
        dir_idx = 0

        while len(order) < total:
            for _ in range(2):
                dr, dc = dirs[dir_idx]

                for _ in range(step_len):
                    r += dr
                    c += dc

                    if 0 <= r < n_rows and 0 <= c < n_cols and not visited[r][c]:
                        visited[r][c] = True
                        order.append(r * n_cols + c)

                        if len(order) == total:
                            break

                dir_idx = (dir_idx + 1) % 4

                if len(order) == total:
                    break

            step_len += 1

        # Страховка, если какая-то клетка не покрылась спиралью
        if len(order) < total:
            remaining = []

            for i in range(n_rows):
                for j in range(n_cols):
                    if not visited[i][j]:
                        remaining.append((i, j))

            remaining.sort(
                key=lambda rc: (
                    (rc[0] - start_r) ** 2 + (rc[1] - start_c) ** 2,
                    rc[0],
                    rc[1]
                )
            )

            for i, j in remaining:
                order.append(i * n_cols + j)

        return order

    def build(self, input_shape):
        h = int(input_shape[1])
        w = int(input_shape[2])
        c = int(input_shape[3])

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError("Height and width must be divisible by patch_size.")

        self.n_rows = h // self.patch_size
        self.n_cols = w // self.patch_size
        self.n_patches = self.n_rows * self.n_cols
        self.features_per_patch = self.patch_size * self.patch_size * c

        order = self._spiral_order(self.n_rows, self.n_cols)
        self.order_tensor = tf.constant(order, dtype=tf.int32)

        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, H, W, C)

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        # patches: (batch_size, n_rows, n_cols, patch_h * patch_w * C)

        batch_size = tf.shape(patches)[0]

        patches = tf.reshape(
            patches,
            [batch_size, self.n_patches, self.features_per_patch]
        )

        # Переставляем патчи в спиральный порядок
        patches = tf.gather(patches, self.order_tensor, axis=1)

        return patches

    def compute_output_shape(self, input_shape):
        h = int(input_shape[1])
        w = int(input_shape[2])
        c = int(input_shape[3])

        n_patches = (h // self.patch_size) * (w // self.patch_size)
        features_per_patch = self.patch_size * self.patch_size * c

        return (input_shape[0], n_patches, features_per_patch)

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
