import numpy

num = 100
def pcc_calculation(out_image, yuantu_image):
    pcc_all = 0
    for i in range(num):
        a = out_image[i].cpu().detach().numpy().squeeze()
        b = yuantu_image[i].cpu().detach().numpy().squeeze()

        a_f = a.flatten()
        b_f = b.flatten()
        pcc_i = numpy.corrcoef(a_f, b_f)[0][1]
        pcc_all += pcc_i
    pcc_mean = pcc_all / num
    return pcc_mean


num_t = 100
def pcc_calculation_t(out_image, yuantu_image):
    pcc_all = 0
    for i in range(num_t):
        a = out_image[i].cpu().detach().numpy().squeeze()
        b = yuantu_image[i].cpu().detach().numpy().squeeze()

        a_f = a.flatten()
        b_f = b.flatten()
        pcc_i = numpy.corrcoef(a_f, b_f)[0][1]
        pcc_all += pcc_i
    pcc_mean = pcc_all / num_t
    return pcc_mean
