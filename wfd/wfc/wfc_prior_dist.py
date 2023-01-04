import numpy as np, os, re, json, sys, time, shutil, heapq

class WFCGeneratorPriorDist():
    def __init__(self, map_size, prior, connection_probs = None,
                 use_uniform_probs = False,
                 boundary_grids = None,
                 preeliminate_border = True,
                 max_saved_states = 3):
        self.prior = prior
        self.map_size = map_size
        connections = connection_probs > 0
        self.tile_count = connections.shape[0]
        self.max_x, self.max_y = map_size
        self.map_allowed_tiles = np.ones((map_size[1], map_size[0], self.tile_count), dtype=bool)
        self.connections = connections
        self.connection_probs = connection_probs
        
        self.use_uniform_probs = use_uniform_probs
        
        self.tile_range = np.arange(self.tile_count)
        self.uniform_probs = np.full((self.tile_count,), 1/self.tile_count)
        self.info_prop_count = 0
        
        self.directions = ((-1, 0), (0, -1), (1, 0), (0, 1))
        
        self.saved_states = []
        self.max_saved_states = max_saved_states
        
        if preeliminate_border:
            self.preeliminate_border_tiles()
        
        self.allow_all_connections_to_null()
        
        self.debug_item = None
        
        if boundary_grids is None:
            self.boundary_grids = np.zeros((map_size[1], map_size[0]), dtype=bool)
        else:
            self.boundary_grids = boundary_grids
    
    def is_grid_allowed(self, grid):
        y, x = grid
        return x >= 0 and y >= 0 and x < self.max_x and y < self.max_y and not self.boundary_grids[grid]
        
    def get_probs_from_map(self, map_tiles):
        # this is the size of the reference map
        max_x = map_tiles.shape[1]
        max_y = map_tiles.shape[0]
        
        tile_count = np.max(map_tiles) + 1
        tile_counts = np.zeros((tile_count,), dtype = np.float32)
        connection_counts = np.zeros((tile_count, 4, tile_count), dtype = np.float32)
        for y, mt_line in enumerate(map_tiles):
            for x, v in enumerate(mt_line):
                if v == 0:
                    continue
                tile_counts[v] += 1
                if x > 0:
                    next_grid = y, x-1
                    nv = map_tiles[next_grid]
                    if nv > 0:
                        connection_counts[v, 0, nv] += 1

                if y > 0:
                    next_grid = y-1, x
                    nv = map_tiles[next_grid]
                    if nv > 0:
                        connection_counts[v, 1, nv] += 1

                if x < max_x - 1:
                    next_grid = y, x+1
                    nv = map_tiles[next_grid]
                    if nv > 0:
                        connection_counts[v, 2, nv] += 1

                if y < max_y - 1:
                    next_grid = y+1, x
                    nv = map_tiles[next_grid]
                    if nv > 0:
                        connection_counts[v, 3, nv] += 1
                    
        tile_counts /= np.sum(tile_counts)
        connection_counts /= np.maximum(1, connection_counts.sum(axis = 2, keepdims = True))
        return connection_counts, tile_counts

    def is_fixed(self, grid):
        return np.sum(self.map_allowed_tiles[grid]) == 1

    def is_zero(self, grid):
        return np.sum(self.map_allowed_tiles[grid]) == 0

    def get_first_possible_state(self, grid):
        return self.map_allowed_tiles[grid].argmax()

    def get_possible_states(self, grid):
        return np.where(self.map_allowed_tiles[grid])[0]
    
    def offset_coords(self, grid, offset):
        y, x = grid
        y += offset[1]
        x += offset[0]
        return y, x
    
    def entropy(self, vec, eps=1e-6):
        return np.sum(np.maximum(0, -vec * np.log(vec + eps)))

    def entropy_all(self, vec, eps=1e-6):
        return np.sum(np.maximum(0, -vec * np.log(vec + eps)), axis = 2)

    def normalize(self, vec):
        s = np.sum(vec)
        if s == 0 or s == 1:
            return vec
        vec /= s
        return vec
    
    def get_probs(self, grid):
        if self.use_uniform_probs:
            return self.uniform_probs

        probs = self.map_allowed_tiles[grid] * self.prior[grid]
        return self.normalize(probs)
    
    def grid_dist(self, g1, g2):
        return abs(g1[0] - g2[0]) + abs(g1[1] + g2[1])
    
    def gather_information(self, grid):
        y, x = grid
        for i, (dx, dy) in enumerate(self.directions):
            next_grid = y - dy, x - dx
            if self.is_grid_allowed(next_grid):
                borders = [False] * 4
                borders[i] = True
                self.propagate_information(next_grid, depth = 1, borders = borders, verbose = False)
    
    def propagate_information(self, main_grid, depth = 2, borders = [True, True, True, True], updated_tiles = None, verbose = True):
        if updated_tiles is None and depth > 1:
            updated_tiles = np.zeros(self.map_allowed_tiles.shape[:2], dtype = bool)
        if depth <= 0:
            return True
        
        if verbose:
            print("prop info at grid {}".format(main_grid))
            
        hp = []
        heapq.heappush(hp, (0, 0, main_grid))
        while len(hp) > 0:
            pri, loop_depth, picked_grid = heapq.heappop(hp)
            y, x = picked_grid
            
            if depth > 1:
                updated_tiles[picked_grid] = False
            state_changed = [None] * 4

            picked_grid_states = self.get_possible_states(picked_grid)
            if len(picked_grid_states) == 0:
                return False
            
            self.info_prop_count += 1
            for i, (dx, dy) in enumerate(self.directions):
                next_grid = y + dy, x + dx
                if self.is_grid_allowed(next_grid) and borders[i]:
                    connectable_all = self.connections[picked_grid_states, i]
                    unconnectable_top = (np.sum(connectable_all, axis = 0) == 0)
                    if self.map_allowed_tiles[next_grid][unconnectable_top].any():
                        self.map_allowed_tiles[next_grid][unconnectable_top] = False
                        if self.is_zero(next_grid):
                            return False
                        if loop_depth < depth - 1 and not updated_tiles[next_grid]:
                            pri = np.sum(self.map_allowed_tiles[next_grid])
                            heapq.heappush(hp, (pri, loop_depth + 1, next_grid))
                            updated_tiles[next_grid] = True
                            
        return True

    def double_check(self, check_rect = None, verbose = False):
        if check_rect is None:
            check_rect = (0, 0, self.max_x, self.max_y)
            
        x1, y1, x2, y2 = check_rect
        
        self.push_state((0, 0), 0)
        
        state = True
        for y in range(y1, y2):
            for x in range(x1, x2):
                grid = y, x
                if self.is_grid_allowed(grid):
                    res = self.propagate_information(grid, depth = 1, verbose = False)
                    if res == False:
                        state = False
                    
        if state == False:
            self.pop_state()
            if verbose:
                print("double check info_prop created zero grids")
            return state
        
        self.discard_state()
        
        area_slice = self.get_slice_from_rect(check_rect)
        summed = np.sum(self.map_allowed_tiles[area_slice], axis = 2)
        if verbose and not (summed == 1).all():
            print("double check not summed to 1")
        return (summed == 1).all()
    
    def allow_all_connections_to_null(self):
        self.connections[0, :, :] = True
        self.connections[:, :, 0] = True
        
        # don't allow the null tile
        self.map_allowed_tiles[:, :, 0] = False
        
    def preeliminate_border_tiles(self):
        left_border =   np.where(np.sum(self.connections[:, 0, :], axis = 1) == 0)[0]
        top_border =    np.where(np.sum(self.connections[:, 1, :], axis = 1) == 0)[0]
        right_border =  np.where(np.sum(self.connections[:, 2, :], axis = 1) == 0)[0]
        bottom_border = np.where(np.sum(self.connections[:, 3, :], axis = 1) == 0)[0]
        
        self.map_allowed_tiles[:, 1:,  left_border]   = False
        self.map_allowed_tiles[1:, :,  top_border]    = False
        self.map_allowed_tiles[:, :-1, right_border]  = False
        self.map_allowed_tiles[:-1, :, bottom_border] = False
        
    def reopen_zero_grids(self, tile_states, initial_state = None, area_size = 1):
        summed = np.sum(tile_states, axis = 2)
        zero_grids = np.where(summed == 0)

        for k in range(len(zero_grids[0])):
            picked_grid = zero_grids[0][k], zero_grids[1][k]
            y, x = picked_grid
            reset_slice = slice(max(0, y-area_size), y+area_size+1), slice(max(0, x-area_size), x+area_size+1)
            if initial_state is None:
                tile_states[reset_slice][:] = True
                tile_states[reset_slice][:, :, 0] = False
            else:
                tile_states[reset_slice][:] = initial_state[reset_slice][:]

    # reopen zero grids
    def loosen_zero_grids(self, tile_states, offset = None, initial_state = None, area_size = 1):
        summed = np.sum(tile_states, axis = 2)
        zero_grids = np.where(summed == 0)
        
        if len(zero_grids[0]) > 0:
            # set zero grids to null
#             summed = np.sum(tile_states, axis = 2)
#             tile_states[summed == 0, :] = False
#             tile_states[summed == 0, 0] = True
            
            for i, y in enumerate(zero_grids[0]):
                x = zero_grids[1][i]
                
                grid = y, x
                
                rand_order = np.arange(4)
                np.random.shuffle(rand_order)
                
                # allow all states
                tile_states[grid][:] = True
                tile_states[grid][0] = False
                
                # propagate from each direction until it becomes zero
                for k in rand_order:
                    dx, dy = self.directions[k]
                    revert_state = tile_states[grid].copy()
                    next_grid = y - dy, x - dx
                    if offset is None:
                        global_grid_index = next_grid
                    else:
                        global_grid_index = self.offset_coords(next_grid, offset)
                        
                    if self.is_grid_allowed(global_grid_index):
                        next_grid_states = self.get_possible_states(next_grid)
                        if len(next_grid_states) >= 0:
                            connectables = self.connections[next_grid_states, k]
                            unconnectables = (np.sum(connectables, axis = 0) == 0)
                            tile_states[grid][unconnectables] = False
                            
                        if np.sum(tile_states[grid]) == 0:
                            tile_states[grid] = revert_state
                            break
                
                # choose random possible answer
                possible_results = np.where(tile_states[grid])[0]
                result = np.random.choice(possible_results)
                
                # assign to grid
                tile_states[grid][:] = False
                tile_states[grid][result] = True
                
                # disallow prop into this grid
                if offset is None:
                    global_grid_index = grid
                else:
                    global_grid_index = self.offset_coords(grid, offset)
                self.boundary_grids[global_grid_index] = True

            # rest of zero grids set to null
            summed = np.sum(tile_states, axis = 2)
            if np.sum(summed == 0) > 0:
                tile_states[summed == 0, :] = False
                tile_states[summed == 0, 0] = True
                return False
            else:
                return True
            
            return False
            
        return True

    def reset(self):
        self.map_allowed_tiles = np.ones((self.map_size[1], self.map_size[0], self.tile_count), dtype=bool)
        
    def get_slice_from_rect(self, rect):
        if rect is None:
            return slice(None), slice(None)
        x1, y1, x2, y2 = rect
        return slice(y1, y2), slice(x1, x2)
    
    def slice_states(self, states, region, copy = True):
        x1, y1, x2, y2 = region
        sliced = states[y1 : y2, x1 : x2]
        if copy:
            return sliced.copy()
        else:
            return sliced
    
    def set_region(self, states, region):
        x1, y1, x2, y2 = region
        self.map_allowed_tiles[y1 : y2, x1 : x2] = states
        
    def push_state(self, grid, state):
        self.saved_states.append((grid, state, self.map_allowed_tiles.copy()))
        if len(self.saved_states) > self.max_saved_states:
            self.saved_states.pop(0)
            
    def pop_state(self):
        if len(self.saved_states) > 0:
            grid, state, self.map_allowed_tiles = self.saved_states.pop()
            return grid, state
        else:
            print("warning: pop_state failed")
            return (0, 0), 0
            
    def discard_state(self):
        self.saved_states.pop(0)

    def check_zero_states(self, name, mapping_region):
        # this is slow, only use this for debugging
        summed = np.sum(self.map_allowed_tiles, axis = 2)
        total_sum = np.sum(summed)
        zero_states = np.sum(summed == 0)
        
        if zero_states > 0:
            print("zero state at check {}: zs = {}".format(name, np.where(summed == 0)))
            return False
        return True
    
    def recheck_borders(self, region, info_depth = 4, borders = [True, True, True, True], verbose = False): 
        x1, y1, x2, y2 = region
        
        region_slice = self.get_slice_from_rect(region)
        self.map_allowed_tiles[region_slice][:] = True
        self.map_allowed_tiles[region_slice][:, :, 0] = False
        
        final_result = True
        if x1 > 0 and borders[0]:
            for y in range(y1, y2):
                x = x1 - 1
                grid = (y, x)
                if self.is_grid_allowed(grid):
                    result = self.propagate_information(grid, depth = info_depth, borders = [False, False, True, False], verbose = False)
                    final_result = final_result and result
        if y1 > 0 and borders[1]:
            for x in range(x1, x2):
                y = y1 - 1
                grid = (y, x)
                if self.is_grid_allowed(grid):
                    result = self.propagate_information(grid, depth = info_depth, borders = [False, False, False, True], verbose = False)
                    final_result = final_result and result
        if x2 < self.max_x and borders[2]:
            for y in range(y1, y2):
                x = x2 + 1
                grid = (y, x)
                if self.is_grid_allowed(grid):
                    result = self.propagate_information(grid, depth = info_depth, borders = [True, False, False, False], verbose = False)
                    final_result = final_result and result
        if y2 < self.max_y and borders[3]:
            for x in range(x1, x2):
                y = y2 + 1
                grid = (y, x)
                if self.is_grid_allowed(grid):
                    result = self.propagate_information(grid, depth = info_depth, borders = [False, True, False, False], verbose = False)
                    final_result = final_result and result
                
        if verbose:
            print("recheck_borders result: {}".format(final_result))
        return final_result
    
    def set_possible_states(self, states, region = None):
        region_slice = self.get_slice_from_rect(region)
        mapping_region = self.map_allowed_tiles[region_slice]
        
        mapping_region[:, :, :] = False
        mapping_region[:, :, states] = True
        
    def set_tiles(self, tiles, region = None):
        region_slice = self.get_slice_from_rect(region)
        mapping_region = self.map_allowed_tiles[region_slice]
        
        mapping_region[:, :, :] = False
        mapping_region[:, :, 0] = False
        meshgrid_tiles = tuple(list(np.meshgrid(np.arange(tiles.shape[1]), np.arange(tiles.shape[0])))[::-1] + [tiles])
        mapping_region[meshgrid_tiles] = True
        mapping_region[tiles == 0, :] = True
    
    def process_initial_states(self, depth = 4):
        # more than 1 restriction (null)
        set_slice = (np.sum(self.map_allowed_tiles, axis = 2) < self.tile_count - 1).copy()
        set_states = np.where(set_slice)
        shadow_states = False
        copy_states = self.map_allowed_tiles[set_slice].copy()
        for i, y in enumerate(set_states[0]):
            x = set_states[1][i]
            grid = y, x
            result = self.propagate_information(grid, depth = depth, verbose = False)
            if not result and not shadow_states:
                print("warning: initial state produces zero states")
                shadow_states = True
        if shadow_states:
            self.map_allowed_tiles[set_slice] = copy_states
            self.boundary_grids[set_slice] = True

    def generate(self, region = None, info_depth = 4, max_iterations = 200000, stall_wait = 50, do_double_check = True, loosen_restrict = False, gather_after_stall = 10, verbose = True):
        region_slice = self.get_slice_from_rect(region)
            
        mapping_region = self.map_allowed_tiles[region_slice]
        initial_state = mapping_region.copy()
        
        current_iteration = 0
        stalled_iterations = 0
        while current_iteration < max_iterations:
            # sum of elements on axis 2 == possible outcomes on grid
            summed = np.sum(mapping_region, axis = 2)
            
            total_sum = np.sum(summed)
            zero_states = np.sum(summed == 0)
            one_states =  np.sum(summed == 1)

            if verbose:
                print("iteration {}: total states {} left, info prop {}, final states {}".format(
                    current_iteration, total_sum, self.info_prop_count, one_states
                ))
            
            # all sums are either 0 or 1
            if (summed == 1).all():
                if not do_double_check or self.double_check(region, verbose = verbose):
                    if verbose:
                        print("The algorithm successfully converged!")
                    return True
                elif loosen_restrict:
                    result = self.loosen_zero_grids(mapping_region, offset = region[:2], initial_state = initial_state)
                    if result:
                        return True
                    else:
                        if verbose:
                            print("loosen_zero_grids failed!")
                        return False
                else:
                    if verbose:
                        print("The algorithm did not pass double check!")
                    return False
            elif np.logical_or(summed == 0, summed == 1).all():
                # normally should not get here, but it frequently does when the algorithm is buggy
                if verbose:
                    print("The algorithm successfully failed!")
                return False
                
            # pick random grid with max entropy
            if len(summed[summed > 1]) > 0:
                prior_region = self.prior[region_slice]
                probs_region = mapping_region * prior_region
                probs_region /= np.maximum(np.sum(probs_region, axis=2, keepdims=True), 1e-3)
                entropy_mat = self.entropy_all(probs_region)
                entropy_mat[summed <= 1] = 1e9
                max_entropy_grids = np.where(entropy_mat == entropy_mat.min())
                # max_entropy_grids = np.where(summed == summed[summed > 1].min())
            else:
                self.reopen_zero_grids(mapping_region, initial_state = initial_state, area_size = 0)
                continue
            picked_grid_index = np.random.randint(0, len(max_entropy_grids[0]))
            picked_grid = max_entropy_grids[0][picked_grid_index], max_entropy_grids[1][picked_grid_index]
                
            # choose a random category
            if region is not None:
                global_grid_index = self.offset_coords(picked_grid, region[:2])
            else:
                global_grid_index = picked_grid
                
            
            if self.use_uniform_probs:
                possible_grid_states = np.where(mapping_region[picked_grid])[0]
                if possible_grid_states[0] == 0:
                    possible_grid_states = possible_grid_states[1:]
                picked_grid_state = np.random.choice(possible_grid_states)
            else:
                picked_grid_probs = self.get_probs(global_grid_index)
                picked_grid_probs[0] = 0
                picked_grid_state = np.random.choice(self.tile_range, p = picked_grid_probs)
                
            # assign to state
#             temp_original = self.map_allowed_tiles.copy()
            self.push_state(global_grid_index, picked_grid_state)
            
            mapping_region[picked_grid][:] = False
            mapping_region[picked_grid][picked_grid_state] = True
                
            # propagate information
            self.info_prop_count = 0
            if region is not None:
                result = self.propagate_information(global_grid_index, depth = info_depth, verbose = verbose)
            else:
                result = self.propagate_information(picked_grid, depth = info_depth, verbose = verbose)
                
            
            # reset if leads to zero state
            if result == False:
#                 self.map_allowed_tiles = temp_original
                global_grid_index, picked_grid_state = self.pop_state()
                mapping_region = self.map_allowed_tiles[region_slice]
                if verbose:
                    print("reset grid at {}".format(global_grid_index))
                self.map_allowed_tiles[global_grid_index][picked_grid_state] = False
                
                if stalled_iterations == gather_after_stall:
                    self.gather_information(global_grid_index)
                
                if self.is_zero(global_grid_index):
                    if verbose:
                        print("grid is reset to zero grid after resetting!!!")
                    global_grid_index, picked_grid_state = self.pop_state()
                    self.map_allowed_tiles[global_grid_index][picked_grid_state] = False
#                 result = self.propagate_information(picked_grid, depth = info_depth)
#                 if result == False:
#                     self.reopen_zero_grids(mapping_region, initial_state = initial_state, area_size = 0)
                stalled_iterations += 1
            else:
                stalled_iterations = 0
                
            # avoid waiting for too long
            if stalled_iterations >= stall_wait:
                if verbose:
                    print("algorithm fails due to stalled generation")
                return False
            
            # not really necessary at this point?
            #self.reopen_zero_grids(mapping_region, initial_state = initial_state, area_size = 2)
            current_iteration += 1
        
        return False
        
    def generate_by_part(self,
                         split_size = 16,
                         generate_size = None,
                         max_retries = 3,
                         param_scheduler = None,
                         max_iterations = 1000,
                         stall_wait = 50,
                         info_depth = 32,
                         do_double_check = True,
                         loosen_restrict = True,
                         verbose_level = 1,
                         verbose_after_retries = 10):
        total = np.ceil(self.max_x / split_size) * np.ceil(self.max_y / split_size)
        
        if param_scheduler is None:
            param_scheduler = lambda retries: (max_iterations, stall_wait)
            
        if generate_size is None:
            generate_size = split_size
            
        if do_double_check is True:
            do_double_check = max_retries + 1
        if loosen_restrict is True:
            loosen_restrict = max_retries + 1
        
        done = 0
        need_to_recheck = False
        need_to_loosen_restrict = False
        for x in range(0, self.max_x, split_size):
            for y in range(0, self.max_y, split_size):
                region = (x, y, min(self.max_x, x + generate_size), min(self.max_y, y + generate_size))
                if need_to_recheck:
                    border_check_result = self.recheck_borders(region,
                                                               info_depth = info_depth,
                                                               borders = [True, True, False, False],
                                                               verbose = verbose_level > 1)
                    need_to_loosen_restrict = False
                    if not border_check_result:
                        need_to_loosen_restrict = True
                        region_slice = self.get_slice_from_rect(region)
                        self.reopen_zero_grids(self.map_allowed_tiles[region_slice], initial_state = None, area_size = 0)
                    
                init_state = self.get_state().copy()

                retries = 0
                max_iter, stall_wait = param_scheduler(retries)
                gen_result = self.generate(region = region,
                                           info_depth = info_depth,
                                           max_iterations = max_iter,
                                           stall_wait = stall_wait,
                                           do_double_check = do_double_check > 0,
                                           loosen_restrict = loosen_restrict <= 0 or need_to_loosen_restrict,
                                           verbose = verbose_after_retries <= 0)
                
                while not gen_result and retries < max_retries:
                    retries += 1
                    max_iter, stall_wait = param_scheduler(retries)
                    if verbose_level > 0:
                        print("grid ({:3d}, {:3d}): retry {}/{}".format(x, y, retries, max_retries))
                    self.set_state(init_state)
                    init_state = init_state.copy()
                    gen_result = self.generate(region = region,
                                               info_depth = info_depth,
                                               max_iterations = max_iter,
                                               stall_wait = stall_wait,
                                               do_double_check = (retries < do_double_check),
                                               loosen_restrict = (retries >= loosen_restrict or need_to_loosen_restrict),
                                               verbose = verbose_after_retries <= retries)
                    
                    if retries >= loosen_restrict:
                        need_to_recheck = True

#                 if not gen_result:
#                     self.set_state(init_state)
#                     self.set_region(True, region)
                done += 1
                if verbose_level > 0:
                    if not gen_result:
                        print("warning: grid ({:3d}, {:3d}) failed to generate".format(x, y))
                    else:
                        print("grid ({:3d}, {:3d}): {:02.2f}% done".format(x, y, done / total * 100))

    def get_state(self):
        return self.map_allowed_tiles

    def set_state(self, state):
        self.map_allowed_tiles = state

    def get_result(self):
        return self.map_allowed_tiles.reshape(-1, self.tile_count).argmax(axis=1).reshape(self.map_size)
    
    def check_result(self, result, connections):
        max_y, max_x = result.shape
        for i in range(max_y):
            for j in range(max_x):
                state = result[i][j]
                if j > 0:
                    next_state = result[i][j - 1]
                    state_allowed = connections[state][0][next_state]
                    print(i, j, state, next_state, 0, state_allowed > 0)
                if i > 0:
                    next_state = result[i - 1][j]
                    state_allowed = connections[state][1][next_state]
                    print(i, j, state, next_state, 1, state_allowed > 0)