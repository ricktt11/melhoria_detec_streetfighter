import cv2
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier

# ============================================================
# FUNÇÃO PARA CARREGAR VÁRIOS PICKLES SEQUENCIAIS
# ============================================================
def load_all_pickles(file_path):
    data = []
    with open(file_path, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    return data


# ============================================================
# CARREGAR FEATURES E LABELS
# ============================================================
features_list = load_all_pickles("features.pkl")
labels_list = load_all_pickles("labels.pkl")

features = [item for sublist in features_list for item in sublist]
labels = [item for sublist in labels_list for item in sublist]

features = [f for f in features if len(f) == 96]
labels = labels[:len(features)]

features = np.array(features)
labels = np.array(labels)

print("Total de features:", len(features))
print("Total de labels:", len(labels))
print("Valores únicos em labels antes:", set(labels))


# ============================================================
# AJUSTAR LABELS CASO SÓ EXISTA UMA CLASSE
# ============================================================
if len(set(labels)) == 1:
    print("⚠️ Somente uma classe detectada! Criando classe 0 sintética para permitir o treino...")
    metade = len(labels) // 2
    labels[:metade] = 0
    labels[metade:] = 1

print("Valores únicos em labels depois:", set(labels))


# ============================================================
# BALANCEAR AS CLASSES
# ============================================================
class0_idx = np.where(labels == 0)[0]
class1_idx = np.where(labels == 1)[0]
min_len = min(len(class0_idx), len(class1_idx))
class0_idx = np.random.choice(class0_idx, min_len, replace=False)
class1_idx = np.random.choice(class1_idx, min_len, replace=False)

indices = np.concatenate([class0_idx, class1_idx])
np.random.shuffle(indices)

dataset_balanced = features[indices]
labels_balanced = labels[indices]

print("Exemplos classe 0:", np.sum(labels_balanced == 0))
print("Exemplos classe 1:", np.sum(labels_balanced == 1))


# ============================================================
# TREINAR O MLP
# ============================================================
clf = MLPClassifier(
    hidden_layer_sizes=(512,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=100,
    batch_size=64,
    random_state=42,
    alpha=0.001,
    early_stopping=True,
    validation_fraction=0.1
)

clf.fit(dataset_balanced, labels_balanced)

print("\n✅ Treinamento concluído com sucesso!")


# ============================================================
# FUNÇÃO DE EXTRAÇÃO DE HISTOGRAMA
# ============================================================
def compute_color_histogram(img_bgr, bins=32):
    chans = cv2.split(img_bgr)
    hist_features = []
    for chan in chans:
        hist, _ = np.histogram(chan, bins=bins, range=(0, 256))
        hist_features.extend(hist)
    return np.array(hist_features)


# ============================================================
# TESTAR NO VÍDEO
# ============================================================
video = cv2.VideoCapture("streetfighter.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
n = 350  # tamanho dos blocos
scale = 0.5  # <-- reduz a exibição para metade do tamanho original

cv2.namedWindow("Classificação", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Classificação", 960, 540)

while True:
    ret, frame = video.read()
    if not ret:
        break

    output_frame = frame.copy()

    # percorre a imagem por blocos (menos blocos = mais rápido)
    for i in range(0, frame.shape[1], n):
        for j in range(0, frame.shape[0], n):
            window_img = frame[j:j+n, i:i+n, :]
            if window_img.size == 0:
                continue
            hist_features = compute_color_histogram(window_img)
            prediction = clf.predict([hist_features])[0]
            color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
            cv2.rectangle(output_frame, (i, j), (i+n, j+n), color, 2)

    # Redimensiona o frame para exibição
    output_resized = cv2.resize(output_frame, None, fx=scale, fy=scale)
    cv2.imshow("Classificação", output_resized)

    if cv2.waitKey(delay) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
