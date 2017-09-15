import numpy as np
import os
import tensorflow as tf

def save_model_params(sess, filename, model_scopes):
    import shelve
    import re

    shelf_data = shelve.open(filename)
    
    for scope_name,scope_vars in model_scopes.iteritems():
        
        print('Saving %s' % (scope_name))
        
        tmp_dict = dict()
        for v in scope_vars:
            tmp_dict[v.name] = sess.run(v)
            
        shelf_data[scope_name] = tmp_dict
        
    print('Model variables saved successfully.')
        
def restore_model_params(sess, filename, model_scopes = []):
    import shelve
    import re

    shelf_data = shelve.open(filename)
    
    if model_scopes==[]:
        model_scopes = shelf_data.keys()
        
    for scope_name in model_scopes:
        
        print('Restoring %s' % (scope_name))
        
        scope_vars = shelf_data[scope_name]
        
        with tf.variable_scope(scope_name, reuse=True) as scope:
            
            for var_name,var_val in scope_vars.iteritems():
                
                l = re.split('/|:',var_name)
                var_name = '/'.join(l[1:-1])

                v = tf.get_variable(var_name)

                assign_op = v.assign(var_val)

                sess.run(assign_op)
                
    print('All models successfully restored.')
