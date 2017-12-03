import os, gzip, PIL, trainModel
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#image in flask to html page: https://stackoverflow.com/questions/11262518/how-to-pass-uploaded-image-to-template-html-in-flask
#using another .py file: https://stackoverflow.com/questions/13034496/using-global-variables-between-files
# reading bytes from image: https://stackoverflow.com/questions/6787233/python-how-to-read-bytes-from-file-and-save-it

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if request.files['file'].filename == '':
            typeError = 'No selected file'
            return render_template('index.html', error = typeError)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            typeError = 'No selected file'
            return render_template('index.html', error = typeError)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            #creates path by combining folder with filename
            filepath = "static/" + filename

            #sets width of image conversion
            basewidth = 28
            #open image
            img = Image.open(filepath)
            
            #set of calculations to convert from current iamge size to 28x28
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

            # convert image to black and white
            img = img.convert('L') 

            #all images are overwritted by the previous as "resized_image.png"
            img.save('static/resized_image.png')

            #this file reader has same intention as reader in trainModel
            with open('static/resized_image.png', 'rb') as f:           
                images = []
                
                for i in range(1):
                    row = []
                    for k in range(28):
                        col = []
                        for j in range(28):
                            col.append(int.from_bytes(f.read(1), "big"))
                        row.extend(col)
                    images.append(row)
            #calls function in trainModel and feeds it image read
            predNum = trainModel.setImage(images)
            #returned number from setImage to be printed to screen
            true_number = "Image predicted to be a: " + str(predNum)

            #returns html page with values to present image and predicated value int convereted to string
            return render_template('index.html', number_name = filename, number = true_number))
    return render_template('index.html')