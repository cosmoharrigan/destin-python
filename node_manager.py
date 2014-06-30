# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import numpy as np
import clustering
import scipy.io as io
import pickle
import cifar as mil
import socket
import random as pyrand
import os
import datetime


MAT_PATH = '/home/syoung22/features.mat'


class FourToOneImage:
    """
    This class defines the structure of the DeSTIN hierarchy.
    """
    
    def __init__(self, channels=3, global_pooling=False, split_quads=False, move_type='Exhaustive', image_size=32, label_size=10):
        # Define Number of Layers

        self.global_pooling = global_pooling
        self.NCHANNELS = channels
        self.LAYERS = 3
        self.CENTS_BY_LAYER = [25, 18, 25]
        self.MEAN_RATE = 0.01
        self.VAR_RATE = 0.01
        self.STARV_RATE = 0.001
        self.WINDOW_WIDTH = 4
        self.VIEW_WIDTH = self.WINDOW_WIDTH * pow(2, self.LAYERS-1)
        self.INPUT_SIZE = self.VIEW_WIDTH * self.VIEW_WIDTH
        self.IMAGE_WIDTH = image_size

        print np.arange(self.LAYERS)
        print np.array(self.CENTS_BY_LAYER)
        print pow(4,np.arange(self.LAYERS))
        layer_belief_sz =  pow(4,np.arange(self.LAYERS)) * np.array(self.CENTS_BY_LAYER)
        self.TOTAL_BELIEF_SIZE = sum(layer_belief_sz)
    

        # Build Structure
        self.struct = []
        self.struct.append(range(pow(4, self.LAYERS - 1)))
        for lay in range(1, self.LAYERS):
            beg_id = self.struct[lay-1][-1]+1
            end_id = pow(4, self.LAYERS-1-lay) + beg_id
            self.struct.append(range(beg_id, end_id))

        # Define Connections and Centroids
        self.conns = []
        self.cents = []
        self.dims = []
        self.mr = []
        self.vr = []
        self.sr = []
        for lay in range(self.LAYERS):
            for node in self.struct[lay]:
                if (lay == 0):
                    self.conns.append([])
                    self.dims.append(self.WINDOW_WIDTH * self.WINDOW_WIDTH * self.NCHANNELS)
                else:
                    offset = self.struct[lay][0]
                    local_id = node - offset
                    width = pow(2, self.LAYERS-lay-1)
                    
                    row = local_id / width
                    col = local_id % width
                    base = [0,1]
                    
                    child_offset = self.struct[lay-1][0]
                    rowOne = [x + 2*col + 4*width*row + child_offset for x in base]
                    rowTwo = [x + 2*width for x in rowOne]
                    self.conns.append(rowOne + rowTwo)
                    self.dims.append(sum([self.cents[x] for x in self.conns[node]]))

                self.cents.append(self.CENTS_BY_LAYER[lay])
                self.mr.append(self.MEAN_RATE)
                self.vr.append(self.VAR_RATE)
                self.sr.append(self.STARV_RATE)

        # Define Viewing Window For Bottom Layer
        # TODO: BETTER VARIABLE NAMES
        self.view =[]
        for node in self.struct[0]:
            base = range(self.WINDOW_WIDTH)
            width = pow(2, self.LAYERS-1)
            node_row = node / width
            node_col = node % width
            single_node_view = []
            for row in range(self.WINDOW_WIDTH):
                single_node_view += [x + self.WINDOW_WIDTH*node_col + row*self.VIEW_WIDTH + node_row*self.VIEW_WIDTH for x in base]

            self.view.append(single_node_view)

        # Define Movements
        move_width = self.IMAGE_WIDTH - self.VIEW_WIDTH + 1
        if move_type == 'Exhaustive':
            temp = np.arange(move_width*move_width*2)
            temp = temp / 2
            temp.shape = (move_width*move_width, 2)
            temp[:,0] = temp[:,0] / move_width
            temp[:,1] = temp[:,1] % move_width
            self.moves = temp.tolist()

        elif move_type == 'Zmoves':
            temp = np.zeros((move_width*3-2,2))
            temp[0:move_width,1] = np.arange(move_width)
            temp[move_width:move_width*2-1,0] = np.arange(1,move_width)
            temp[move_width:move_width*2-1,1] = np.arange(move_width-2,-1,-1)
            temp[move_width*2-1:move_width*3-2,0] += move_width - 1
            temp[move_width*2-1:move_width*3-2,1] = np.arange(1,move_width)
            self.moves = temp.tolist()

        self.num_moves = len(self.moves)

        if self.global_pooling:
            self.moves_to_save = [1,]
        else:
            self.moves_to_save = [9, 13, 17] #updated for zero index

        self.label_size = label_size # This is the number of classes.

        if split_quads:
            center = int(move_width)/2 + 1
            self.quad_idx = 0 + (temp[:,0] < center) + (temp[:,1] < center)*2
            self.quad_idx.shape = (self.num_moves,)
            self.quad_idx = self.quad_idx.tolist()
            self.belief_blocks = 4
        else:
            self.quad_idx = np.zeros(self.num_moves)
            self.quad_idx = self.quad_idx.tolist()
            self.belief_blocks = 1
    

class NodeManager:
    """
    This class handles all DeSTIN operations.
    """
    
    def __init__(self, params):
        """
        Creates DeSTIN hierarchy based on parameters.
        """
        
        self.params = params
        
        self.hierarchy = []
        for lay in params.struct:
            for node in lay:
                newNode = clustering.SupNode(params.mr[node], params.vr[node],
                                  params.sr[node], params.dims[node],
                                  params.cents[node], node, params.label_size)
                for child in params.conns[node]:
                    newNode.add_child(self.hierarchy[child])
                self.hierarchy.append(newNode)


        self.date_training_started = datetime.datetime.now()


    def process_images(self, images, steps, train, record_beliefs, labels, savefile='', ind_moves=None):
        """
        Provide images to DeSTIN to train/test on.
        """

        print images.shape
        print labels.shape
        count = 0
        if record_beliefs:
            features = np.zeros((images.shape[0], len(self.params.moves_to_save)*self.params.belief_blocks, self.params.TOTAL_BELIEF_SIZE))

        if train:
            rand_idx = np.random.permutation(images.shape[0])
        else:
            rand_idx = range(images.shape[0])

        if not steps:
            steps = images.shape[0]
            
        if ind_moves is not None:
            indfeats = np.zeros((images.shape[0],self.params.num_moves,self.params.TOTAL_BELIEF_SIZE))

        while (count < steps):
            for img in range(0,min(images.shape[0],steps-count)):
                if not (count % 100):
                    if savefile != '':
                        self.save(savefile)
                    print count
                
                curr_img = images[rand_idx[img]]
                curr_label = labels[rand_idx[img]]
                one_hot_label = np.zeros((1,self.params.label_size))
                one_hot_label[0,curr_label.astype(int)] = 1.0

                for move, movnumber in zip(self.params.moves, range(self.params.num_moves)):
                    cropped_image = np.copy(curr_img[move[0]:move[0]+self.params.VIEW_WIDTH,
                                             move[1]:move[1]+self.params.VIEW_WIDTH])#copy necessary for reshape; maybe something better?

                    cropped_image.shape = (1, self.params.INPUT_SIZE, self.params.NCHANNELS)
                    self.process_single_image(cropped_image, train, one_hot_label)

                    if record_beliefs:
                        if self.params.global_pooling: # This should be the default. 
                            div1 = 0
                            for node in self.hierarchy:
                                div2 = div1 + node.belief.size
                                features[rand_idx[img],int(self.params.quad_idx[movnumber]),div1:div2] += np.squeeze(node.belief)
                                if ind_moves is not None:
                                    indfeats[rand_idx[img],movnumber,div1:div2] = np.squeeze(node.belief)
                                div1 = div2
                        else:
                            if movnumber in self.params.moves_to_save:
                                mov_idx = self.params.moves_to_save.index(movnumber)
                                div1 = 0
                                for node in self.hierarchy:
                                    div2 = div1 + node.belief.size
                                    features[rand_idx[img],mov_idx,div1:div2] = node.belief
                                    div1 = div2

                self.clear_states() # Clear belief states between images (only really needed when using rec. clustering)
                count += 1

        if record_beliefs:
            features.shape = (images.shape[0], self.params.belief_blocks*len(self.params.moves_to_save) * self.params.TOTAL_BELIEF_SIZE)
            if ind_moves is not None:
                return (features,indfeats,self.params.moves)
            else:
                return features           

    def process_single_image(self, img, TRAIN, label):
        """
        Provide a single image (an image for the current movement, 
        cropped to the viewing size of the bottom layer) to the DeSTIN hierarchy.
        This should not be confused with the large (e.g. 48x48) images which we will call
        "example images." This refers to the "movement images" (e.g. 16x16).
        """

        for node in self.hierarchy:
            
            if not node.children:
                node.update_node(img[0,self.params.view[node.ID]], TRAIN, label)
            else:
                node.latched_update(TRAIN, label)

    def clear_states(self):
        """
        Clear the belief states. Used between images.
        """

        for node in self.hierarchy:
            node.clear_belief()

    def init_whitening(self, mn=[], st=[], tr=[]):
        """
        Provide whitening parameters to nodes.
        """

        for node in self.hierarchy:
            node.init_whitening(mn=mn, st=st, tr=tr)

    def save(self, filename):
        """
        Save hierarchy for later use or inspection.
        """

        pickle.dump(self, open(filename + '.p', 'wb'))
        
    def limit_moves(self,num_moves=5):
        """
        Restrict moves to those necessary for generating
        beliefs for individual movements.
        """

        self.saved_moves = self.params.moves
        self.saved_num_moves = self.params.num_moves
        
        keep_moves = pyrand.sample(xrange(self.params.num_moves), num_moves)
        
        self.params.moves = [self.params.moves[x] for x in keep_moves]
        self.params.num_moves = num_moves
        
        
    def restore_moves(self):
        """
        Restore moves after using limit_moves
        """

        self.params.moves = self.saved_moves
        self.params.num_moves = self.saved_num_moves

def main():
    """
    Main function that runs DeSTIN on a dataset and saves belief states
    as feature vectors.
    """

    params = FourToOneImage(channels=3, global_pooling=True, split_quads=False)
    nm = NodeManager(params)

    data = mil.load_cifar()
    nm.init_whitening(mn=data['patch_mean'], st=data['patch_std'], tr=data['whiten_mat'])
    
    # Train
    nm.process_images(data['train_data'],15000,True,False,data['train_labels'],'for_viewing_centroids_heat_map')
  
    # Record Beliefs
    features = nm.process_images(data['train_data'],[],False,True,np.zeros(data['train_labels'].shape))#array of zeros, no labels during testing
    vts ={}
    vts['features'] = features
    vts['labels'] = data['train_labels']
 
    # Record Beliefs
    features = nm.process_images(data['test_data'],[],False,True,np.zeros(data['test_labels'].shape))

    vts['test_features'] = features
    vts['test_labels'] = data['test_labels']
    io.savemat(MAT_PATH, vts)

if __name__ == "__main__":
    main()
