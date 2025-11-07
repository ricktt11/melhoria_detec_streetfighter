import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

# ==============================
# CONFIGURAÇÕES
# ==============================
n = 350         # tamanho do bloco
scale = 0.5      # fator de redução da janela
video_path = 'streetfighter.mp4'

# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def compute_histogram(img_bgr, bins=32):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hist_r = np.histogram(img_rgb[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(img_rgb[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(img_rgb[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    hist /= np.sum(hist) + 1e-6
    return hist

def mouse_click(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x, orig_y = int(x / scale), int(y / scale)
        clicked_points.append((orig_x, orig_y, 'left'))  # verde
    elif event == cv2.EVENT_RBUTTONDOWN:
        orig_x, orig_y = int(x / scale), int(y / scale)
        clicked_points.append((orig_x, orig_y, 'right'))  # vermelho

# ==============================
# VARIÁVEIS GLOBAIS
# ==============================
clicked_points = [] 
treined = False

clf = MLPClassifier(
    hidden_layer_sizes=(512,),
    activation="relu",
    solver='adam',
    learning_rate_init=0.001,
    alpha=0.0005,
    batch_size=16,
    max_iter=50,
    warm_start=True,
    shuffle=True,
    random_state=42
)

cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("test", 960, 540)
cv2.setMouseCallback("test", mouse_click)

video = cv2.VideoCapture(video_path)
frame_id = 0

# ==============================
# LOOP PRINCIPAL
# ==============================
while True:
    ret, base_frame = video.read()
    if not ret:
        break

    h, w, _ = base_frame.shape
    frame = base_frame.copy()

    # cria grid padrão (tudo verde = 1)
    dataset = []
    labels = []
    for j in range(0, h, n):
        for i in range(0, w, n):
            window_img = base_frame[j:j + n, i:i + n, :]
            hist_features = compute_histogram(window_img)
            dataset.append(hist_features)
            labels.append(1)  # padrão = verde
            cv2.rectangle(frame, (i, j), (i + n, j + n), (0, 255, 0), 1)

    # loop de interação por frame
    while True:
        show_frame = frame.copy()

        # aplica os cliques (vermelho/verde)
        for x, y, btn in clicked_points:
            x_window = x // n
            y_window = y // n
            idx = y_window * (w // n) + x_window
            if 0 <= idx < len(labels):
                labels[idx] = 0 if btn == 'right' else 1  # direita = 0 (vermelho)
                y1, y2 = y_window * n, (y_window + 1) * n
                x1, x2 = x_window * n, (x_window + 1) * n
                color = (0, 0, 255) if btn == 'right' else (0, 255, 0)
                cv2.rectangle(show_frame, (x1, y1), (x2, y2), color, -1)

        # exibe o frame reduzido
        show_frame_resized = cv2.resize(show_frame, None, fx=scale, fy=scale)
        cv2.imshow("test", show_frame_resized)

        key = cv2.waitKey(1)

        # SALVAR FRAME E AVANÇAR
        if key == ord('n'):
            # aplica cliques antes de salvar
            with open("features.pkl", "ab") as f:
                pickle.dump(dataset, f)
            with open("labels.pkl", "ab") as f:
                pickle.dump(labels, f)
            print(f"Frame {frame_id} salvo com {len(dataset)} blocos.")
            clicked_points = []
            break

        # PULAR FRAME
        if key == ord('p'):
            clicked_points = []
            break

        # TREINAR
        if key == ord('t'):
            with open("features.pkl", 'rb') as f:
                all_features = []
                try:
                    while True:
                        all_features.extend(pickle.load(f))
                except EOFError:
                    pass

            with open("labels.pkl", 'rb') as f:
                all_labels = []
                try:
                    while True:
                        all_labels.extend(pickle.load(f))
                except EOFError:
                    pass

            clf.fit(all_features, all_labels)
            with open("mlp_model.pkl", "wb") as f:
                pickle.dump(clf, f)

            print("Treinamento concluído! Modelo salvo em mlp_model.pkl")
            treined = True
            clicked_points = []

        # SAIR
        if key == 27:
            video.release()
            cv2.destroyAllWindows()
            raise SystemExit

    frame_id += 1

video.release()
cv2.destroyAllWindows()
