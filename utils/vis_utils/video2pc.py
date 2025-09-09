import cv2

video_path = 'data/pexels_videos.mp4'  # 视频地址
output_path = 'data/video2img/'  # 输出文件夹
interval = 1  # 每间隔10帧取一张图片

if __name__ == '__main__':
    num = 1
    vid = cv2.VideoCapture(video_path)
    while vid.isOpened():
        is_read, frame = vid.read()
        if is_read:
            # if num % interval == 1:
            #     file_name = '%06d' % num
            #     cv2.imwrite(output_path + str(file_name) + '.jpg', frame)
            #     # 00000111.jpg 代表第111帧
            #     cv2.waitKey(1)
            # num += 1

            file_name = '%06d' % num
            cv2.imwrite(output_path + str(file_name) + '.jpg', frame)
                # 00000111.jpg 代表第111帧
            cv2.waitKey(1)
            num += 1

        else:
            break
