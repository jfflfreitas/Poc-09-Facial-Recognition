import face_recognition
import cv2

#Obtendo uma referência da webcam
video_capture = cv2.VideoCapture(0)

#Carregando uma imagem de referência e aprendendo a reconhecê-la
john_image = face_recognition.load_image_file("john.jpg")
john_face_encoding = face_recognition.face_encodings(john_image)[0]

#Carregando uma segunda imagem de referência aprendendo a reconhecê-la
akira_image = face_recognition.load_image_file("akira.jpg")
akira_face_encoding = face_recognition.load_image_file(akira_image)[0]

#Criando arrays para reconhecer as face encodings e seus nomes
known_face_encodings = [
    john_face_encoding,
    akira_face_encoding
]
known_face_names = [
    "John Franklin",
    "Claudio akira Endo"
]

#Inicializando algumas variáveis
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    #Capturando um único frame do vídeo
    ret, frame = video_capture.read()

    #Efetuando o resize da imagem para 1/4 para um processamento mais rápudo
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

    #Convertendo a imagem de BRG que é uilizada Opencv para RGB que é o padrão do face_recognition
    rgb_small_frame = small_frame[:, :, ::-1]

    #Processando somente todos os outros frames do vídeo pra economizar tempo
    if process_this_frame:
        #Encontrando todas as faces e face_encodings no frame de video atul
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            #Verificando se a face corresponde a uma das faces conhecidas
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            #Se um "match" foi encontrado em Known_face_encodings, usa-se somente o primeiro
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame

    #Exibindo o resultado
    for(top, right, bottom, left), name in zip(face_locations, face_names):
        # Escalando os locais das faces de backup, pois o quadro detectado foi dimensionado para 1/4 do tamanho
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        #Desenhando uma borda envolta do rosto conhecido
        cv2.rectangle(frame,(left, top), (right, bottom), (0, 0, 255), 2)

        #Desenhando o nome embaixo da imagem
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom -6 ), font, 1.0, (255, 255, 255), 1)

        #Mostrando a imagem resultante
        cv2.imshow('Video', frame)

        #Pressione 'q' no teclado para sair!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #Liberando o controle da webcam
        video_capture.release()
        cv2.destroyAllWindows()