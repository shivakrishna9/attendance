from rpn.generate import imdb_proposals
from datasets.factory import get_imdb

def generate_rpn():

	home = "/home/walle/"

    MODEL_FILE = home + 'Attendance/extras/models/ResNet-101-deploy.prototxt'
    PRETRAINED = home + 'Attendance/extras/models/ResNet-101-model.caffemodel'

    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TRAIN)

	imdb = get_imdb(args.imdb_name)
    imdb_boxes = imdb_proposals(net, imdb)

    output_dir = get_output_dir(imdb, net)
    rpn_file = os.path.join(output_dir, net.name + '_rpn_proposals.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)
    print 'Wrote RPN proposals to {}'.format(rpn_file)


if __name__ == '__main__':
	generate_rpn()
	