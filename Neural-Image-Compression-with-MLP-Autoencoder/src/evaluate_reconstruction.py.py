def activation_fuction(y_in):
    result=1/(1+math.exp(-1*y_in))
    return result

p=4

#ساخت بردار ورودی
def open_file(lenght,pic):
    input_vector=[]
    block_num=256**2/lenght**2
    block_num=int(block_num)
    adress='/content/drive/MyDrive/Q1/TestSet/'+pic+ ".jpg"
    img=cv2.imread(adress,cv2.COLOR_BGR2GRAY)
    for j in range(block_num):
        vector=[]
        vector.append(1)
        start_x=(j//(256/lenght))*(lenght)
        start_x=int(start_x)
        start_y=(j%(256/lenght))*(lenght)
        start_y=int(start_y)
        for h in range(start_x,start_x+lenght):
            for k in range(start_y,start_y+lenght):
                a=float(img[h,k])
                a/=255
                vector.append(a)
        input_vector.append(vector)
    return(input_vector)


#محاسبه خروجی شبکه به ازای بردار های ورودی
def Test(v,w,input_vector,neuron_num):
    #تعیین مقادیر اولیه وزن ها از روی فایلی که از مرحله آموزش به دست آمده است
    final=[]
    test_set_num=len(input_vector)#تعداد داده آموزش
    n=len(input_vector[0])#تعداد نورون ورودی
    p=neuron_num #تعداد نورون لایه مخفی
    m=64 #تعداد نورون خروجی
    for h in range(test_set_num): # h is number of train data
        x=input_vector[h]
        z_in_list=[]
        z_out_list=[]
        for j in range(p):
            z_in=0
            for i in range(n):
                z_in+=float(v[i,j])*x[i]
            z_in_list.append(z_in)
            z_out_list.append(activation_fuction(z_in))
        z_out_list.insert(0,1) # for bias
        y_in_list=[]
        y_out_list=[]
        for k in range(m):
            y_in=0
            for j in range(p+1):
                y_in+=float(w[j,k])*z_out_list[j]
            y_in_list.append(y_in)
            a=activation_fuction(y_in)
            a*=255
            a=round(a)
            y_out_list.append(a)
        final.append(y_out_list)
    return(final)

# نمایش عکس خروجی با چسباندن بلوک های 8 در 8
def show(lenght,output,adress,pic):
    adress='result\\'+adress+'\\'+pic+'.jpg'
    block_num=256**2/lenght**2
    block_num=int(block_num)
    img=np.zeros((256,256,3))
    for j in range(block_num):
        start_x=(j//(256/lenght))*(lenght)
        start_x=int(start_x)
        start_y=(j%(256/lenght))*(lenght)
        start_y=int(start_y)
        block_val=output[j]
        for i in range (lenght**2):
            x=i//8
            y=i%8
            x+=start_x
            y+=start_y
            img[x,y,0]=block_val[i]
            img[x,y,1]=block_val[i]
            img[x,y,2]=block_val[i]
    cv2.imwrite(adress, img)
    return adress

#محاسبه psnr
def psnr_cal(input,output):
    total=0
    dif=0
    for row in range(256):
        for col in range(256):
            dif=float(input[row,col])-float(output[row,col,0])
            dif=dif**2
            total+=dif
    psnr=(255**2)/(1/(row*col)*total)
    psnr=10*math.log10(psnr)
    return psnr


#Main part test
#نام عکس ها
test_name=['camera','crowd','house','lena','pepper']

# آدرس مسیری که فایل‌های وزن در آن قرار دارند
weight_dir = '/content/drive/MyDrive/Q1/weights_for_'+str(p)


# فایل‌های وزن از مسیر
weight_files = os.listdir(weight_dir)

file_v = None
file_w = None

for weight_file in weight_files:
    if weight_file.startswith('v_'):
        file_v = open(os.path.join(weight_dir, weight_file), 'r')
    elif weight_file.startswith('w_'):
        file_w = open(os.path.join(weight_dir, weight_file), 'r')

v = []
w = []

if file_v is not None:
    for line in file_v.readlines():
        stripped_line = line.strip('\n')
        line_list = list(map(float, stripped_line.split(',')))
        v.append(line_list)
    file_v.close()

if file_w is not None:
    for line in file_w.readlines():
        stripped_line = line.strip('\n')
        line_list = list(map(float, stripped_line.split(',')))
        w.append(line_list)
    file_w.close()

    v = np.array(v)
    w = np.array(w)

    print(f'\nResults for {p} hidden layer neurons:\n')

    total_psnr = 0
    for pic in test_name:
        input = open_file(8, pic)
        output = Test(v, w, input, p)
        adress = show(8, output, str(p), pic)

        img_output = cv2.imread(adress, cv2.COLOR_BGR2GRAY)
        adress = '/content/drive/MyDrive/Q1/TestSet/' + pic + '.jpg'
        img_input = cv2.imread(adress, cv2.COLOR_BGR2GRAY)
        psnr = psnr_cal(img_input, img_output)

        total_psnr += psnr

        # same dimensions for input and output image
        if img_input.ndim == 3 and img_input.shape[2] == 3:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

        if img_output.ndim == 3 and img_output.shape[2] == 3:
            img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)

        # resize and concatenate them side by side
        concatenated_img = np.hstack((img_input, img_output))

        output_path = '/content/drive/MyDrive/Q1/Results/'+str(p)+'/'+str(p)+'_'+pic+'_concatenated_image.jpg'

        cv2.imwrite(output_path, concatenated_img)

        print(f'PSNR for {pic} = {psnr}')

    avg_psnr = total_psnr / len(test_name)
    print('Concatenation and saving of test images completed.')
    print(f'\nAvg PSNR = {avg_psnr}')