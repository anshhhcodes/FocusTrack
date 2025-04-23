import cv2
import time

# def main():
#     cap = cv2.VideoCapture(0)
#     frame_rate = 3
#     prev = 0

#     while (True):

#         time_elapsed = time.time() - prev
#         res, image = cap.read()

#         if time_elapsed > 1. / frame_rate:
#             prev = time.time()

#             cv2.imshow('frame', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#             # Do something with your image here.

def main():
    cap = cv2.VideoCapture(0)
    prev = 0

    while True:
        # Capture frame-by-frame
        res, image = cap.read()

        # Calculate FPS
        time_elapsed = time.time() - prev
        if time_elapsed > 0:
            fps = 1. / time_elapsed
        else:
            fps = 0

        # Update the previous time
        prev = time.time()

        # Display the FPS on the frame
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('frame', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()