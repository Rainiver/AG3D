import cv2

video_path = 'data/pexels_videos.mp4'  # Path to the video file
output_path = 'data/video2img/'        # Output folder for extracted frames
interval = 1                           # Extract one frame every N frames (here set to 1)

if __name__ == '__main__':
    num = 1
    vid = cv2.VideoCapture(video_path)
    while vid.isOpened():
        is_read, frame = vid.read()
        if is_read:
            # Example: extract one frame every "interval" frames
            # if num % interval == 1:
            #     file_name = '%06d' % num
            #     cv2.imwrite(output_path + str(file_name) + '.jpg', frame)
            #     # 00000111.jpg means this image corresponds to frame 111
            #     cv2.waitKey(1)
            # num += 1

            file_name = '%06d' % num
            cv2.imwrite(output_path + str(file_name) + '.jpg', frame)
            # 00000111.jpg means this image corresponds to frame 111
            cv2.waitKey(1)
            num += 1

        else:
            break

