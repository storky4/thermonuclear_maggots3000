import cv2
import face_recognition
import pickle
with open('data.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)
video_capture = cv2.VideoCapture(0)
while True:
    # Получаем кадр из веб-камеры
    ret, frame = video_capture.read()
    # Преобразуем цветовое пространство из BGR в RGB
    rgb_frame = frame[:, :, ::-1]
    # Находим все лица на кадре
    face_locations = face_recognition.face_locations(rgb_frame)
    # Если нашли хотя бы одно лицо
    if len(face_locations) > 0:
        # Кодируем найденные лица
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        # Для каждого найденного лица
        for face_encoding in face_encodings:
            # Сравниваем его с кодировками из базы данных
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            # Находим индекс первого совпадения
            match_index = next((i for i, match in enumerate(matches) if match), None)
            if match_index is not None:
                # Выводим имя человека, если его данные есть в БД
                print(known_face_names[match_index])
            else:
                # Запрашиваем у пользователя имя и добавляем его в БД
                new_name = input("Введите имя: ")
                known_face_encodings.append(face_encoding)
                known_face_names.append(new_name)
                with open('data.pkl', 'wb') as f:
                    pickle.dump((known_face_encodings, known_face_names), f)
video_capture.release()
cv2.destroyAllWindows()
