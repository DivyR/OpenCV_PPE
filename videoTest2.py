import cv2

testCascPath = "haarcascades/"+"haarcascade_frontalface_default.xml"

palmCascPath = "haarcascades/"+"palm.xml" #5, 60,60
eyePairBigPath = "haarcascades/"+"eyePairBig.xml" #6, 60,60
mouthCascPath = "haarcascades/"+"mouth.xml" #40, 60,60
upperBodyCascPath = "haarcascades/"+"upperBody.xml" #20, 60,60
noseCascPath = "haarcascades/"+"nose.xml" #80, 60,60
fullBodyCascPath = "haarcascades/"+"fullBody.xml" #1, 60,60
faceCascPath = "haarcascades/" + "haarcascade_frontalface_default.xml"
fistCascPath = "haarcascades/" + "fist.xml"


# Create the haar cascades
testCascade = cv2.CascadeClassifier(testCascPath)

palmCascade = cv2.CascadeClassifier(palmCascPath)
eyePairCascade = cv2.CascadeClassifier(eyePairBigPath)
mouthCascade = cv2.CascadeClassifier(mouthCascPath)
upperBodyCascade = cv2.CascadeClassifier(upperBodyCascPath)
noseCascade = cv2.CascadeClassifier(noseCascPath)
fullBodyCascade = cv2.CascadeClassifier(fullBodyCascPath)
faceCascade = cv2.CascadeClassifier(faceCascPath)
fistCascade = cv2.CascadeClassifier(fistCascPath)

cascs = [palmCascade,eyePairCascade,mouthCascade,upperBodyCascade,noseCascade,fullBodyCascade,faceCascade,fistCascade]

#list of minNeighbors
minNeighbors = [75,2,40,30,80,1,40,75]  #92,

#list of colors
colors = [
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0),
    (0,255,255),
    (255,0,255),
    (255,255,255),
    (120,50,200)
]

for i in range(1,50,1):
    imagePath = "vidImages/image"+str(i)+".jpg"

    # Read the image
    image = cv2.imread(imagePath)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    for i in range(0,len(cascs)):
        rects = cascs[i].detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=minNeighbors[i],
            minSize=(60,60)
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x+w, y+h), colors[i], 2)


    scale_percent = 40 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("Found", resized)
    cv2.waitKey(0)