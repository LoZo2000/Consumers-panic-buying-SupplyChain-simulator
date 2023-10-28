from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import SingleGrid
import numpy as np
import random
import math as m
import matplotlib.pyplot as plt



"""System parameters"""
DC_starting_inventory = 10000
m_starting_capacity = 10000           
wh_region = 100                         # Width & Height of a region
w_grid = wh_region*2 #5                 # Full width of the grid
h_grid = wh_region*3 #4                 # Full height of the grid
N_regions = int(w_grid/wh_region)*int(h_grid/wh_region) # Number of regions
f_restock = 1/3                         # Restock frequency [/day]
m_transport_time = 2                    # Manufacturer-to-DC transport time [days]
dc_transport_time = 1                   # DC-to-store transport time [days]

# Number of agents
N_m = 1                                 # Number of Manufacturers 
N_dc = 1                                # Number of distribution centers 
N_s = 5*N_regions                       # Number of stores 
N = 300*N_s                             # Number of customers          
customer_store_limit = N_s              # Limit of stores a customer will visit to satisfy the purchase 
p_mult = 2                              # Multiplying factor for the increase in buying frequency and quantity of a customer in panic

"""Consumption behavior of customers (equal for each region)"""
consumption_list_region = np.zeros([int(N/N_regions),2])
consumption_list_region[range(int(0.27*N/N_regions))] = [0.333, int(1)]
consumption_list_region[range(int(0.27*N/N_regions),int(0.88*N/N_regions))] = [0.143, 3]
consumption_list_region[range(int(0.88*N/N_regions),int(0.98*N/N_regions))] = [0.067, 6]
consumption_list_region[range(int(0.98*N/N_regions),int(N/N_regions))] = [0.033, 11]
consumption_list_region = consumption_list_region.tolist() #convert a given array to an ordinary list with the same items, elements, or values



# Initialize live graph:
plt.ion()
class ResponsiveGraph():
    def __init__(self, t):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3)
        self.fig.set_figwidth(20) 
        self.fig.set_figheight(9)

        # Map chart
        self.ax1.set_title("Map")
        self.agents_in_cell = np.zeros((h_grid, w_grid))
        self.ax1.imshow(self.agents_in_cell, interpolation='nearest', cmap = 'Reds_r')

        # Demand chart
        self.ax2.set_title("Demand")
        self.t = range(t)
        self.demand = np.zeros(t)
        self.av_demand = np.zeros(t)
        
        # IFR chart
        self.ax3.set_title("Item Fill Rate")
        self.ifr_dc = np.zeros(t)
        self.av_ifr_dc = np.zeros(t)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, r, d, ifr):
        plt.cla()
        
        # Map chart
        self.ax1.imshow(self.agents_in_cell, interpolation='nearest', cmap = 'Reds_r')
        
        # Demand chart
        self.demand[r] = d
        self.av_demand[r] = demand[:r].mean()
        self.ax2.plot(self.t[:r], self.demand[:r], 'k', marker='o')
        self.ax2.plot(self.t[:r], self.av_demand[:r], 'k--')
        
        # IFR chart
        self.ifr_dc[r] = ifr
        self.ax3.plot(self.t[:r], self.ifr_dc[:r], 'b', marker='o')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    


"""Agent classes definition"""
class Customer(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.q_base = 0             # Customer's regular purchasing quantity 
        self.f_base = 0             # Customer's regular purchasing frequency
        self.purchasing_store = 0   # Store in which the customer makes a purchase
        self.q = self.q_base        # Customer's current purchasing quantity
        self.f = self.f_base        # Customer's current purchasing frequency
        self.stores = []            # List of all stores. The list determines the sequence according to which the Customer visits the stores
        self.pbs = False            # Customer's panic buying sensitivity
        self.hungry = False         # If the customer doesn't manage to find a store with enough pasta, it gets hunger
    
    def store_visiting_list_generation(self): 
        """The sequence of stores the Customer can visit is retrieved"""
        s_list = np.array(self.model.schedule.agents[N:N+N_s])      # the Store agents are retrieved from the schedule.agents list of the scheduler
        self.stores = s_list
        self.purchasing_store=random.choice(self.stores) # the Customer chooses randomly a Store in the list of Store agents
        self.purchasing_store.store_expected_demand += self.q_base*self.f_base   # the Customer's expected daily demand is recorded for the Store agent chosen by the Customer. This information is used to set the different Stores' starting inventory. 
        
    def initial_panic(self):
        if self in self.model.initial_panic_customers:  # If, in the current step (day), the Customer is a panic buyer, 
            self.pbs=True                               # his sensitivity to panic buying is recorded,
            self.f = self.f*p_mult                      # his buying frequency is increased,
            self.q = self.q*p_mult                      # his buying quantity is increased.
                       
        else:                                           # If, in the current step (day), the Customer is not a panic buyer,
            self.pbs=False                              # his non-sensitivity to panic buying is recorded, 
            self.f = self.f_base                        # his purchasing frequency is set to the base value,
            self.q = self.q_base                        # his purchasing quantity is set to the base value.
            
    def purchase(self):
           """ The Store at which the Customer will purchase is identified"""
           if i != 0: # During the first step (day), it is assumed that the store chosen for the purchase has inventory 
               """The Customer looks for a store with inventory to make a purchase"""
               for n in [*range(customer_store_limit)]:
                   store=random.choice(self.stores) # the Customer chooses randomly the Store he is going to visit in the step
                   while(store==self.purchasing_store): # but if he has already tried to make the purchase at that Store but he found no inventory, 
                       store=random.choice(self.stores) # he chooses another Store, 
                   self.purchasing_store=store          # If he has not already tried to purchase at the Store in the step (day), he tries to make the purchase at the chosen Store. 
                   if self.purchasing_store.store_inv > 0: # The Customer checks whether there is inventory left at the chosen store 
                      break
                          
           """The Customer makes the purchase at the store he has just selected"""
           if  self.q > self.purchasing_store.store_inv and i != 0:           # If the selected store does not have the entire quantity requested by the Customer,
               p_amount = self.purchasing_store.store_inv                    # the Customer will purchase an amount equal to the reamining inventory at the store. In case the store is in stock out, such amount will be 0 (as the store will have no inventory left)
               self.hungry = True
           else:                                             # If the selected store has enough inventory to fulfill the entire quantity requested by the Customer,
               p_amount = self.q                             # the Customer will purchase an amount that is equal to his purchasing quantity
               self.hungry = False
               '''Following the purchase, inventory and demand attributes are updated'''
           self.purchasing_store.store_inv -= p_amount # Once the Customer has made the purchase, the inventory at the store is decreased of the purchased amount.
           self.purchasing_store.d_s[-1] += p_amount   # the purchased amount by the Customer is added to the amount of products bought at the store in the step (day).
           self.model.d += self.q      # the quantity requested by the Customer is added to the overall demand in the step (day)
   
    def step(self):
        """The list of stores the Customer can visit is retrieved and the initial expected demand is recorded (for the Stores to set their starting inventory)"""
        if i==0:
            self.store_visiting_list_generation()   
        """Starting from the beginning of the pandemic, until panic Customers are generated, assign them a raised purchasing behavior and panic sensitivity"""
        if self.model.pandemic_count >= 1 and i<panic_end:
            self.initial_panic()
        """During the first step (day) in which the % of panic Customers goes back to 0, make sure that the purchasing behavior and panic sensitivity of all Customers is back to base values"""
        if i==panic_end: 
            self.pbs=False
            self.f = self.f_base
            self.q = self.q_base
        """Determine whether the Customer will buy in the step"""
        """The Customer has a chance of self.f of purchasing in the step (day)"""
        if random.choice(range(0,int(1e3))) < int(1e3)*self.f:       # The method random.choice draws a random integer between 0 and 1000; the randomly drawn number has probability self.f of being less than self.f multiplied by 1000 (self.f is multiplied by 1000 to obtain an integer value and make it comparable to the randomly drawn number). Therefore, if the randomly drawn number is less than self.f *1000,  
            self.purchase()                                          # the Customer will purchase. 
            self.purchasing_store=0                                    # Resets the last Store at which the Customer has made his purchase so that, potentially, the Store could be chosen again the next time the Customer purchases
                    
class Store(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.store_expected_demand = 0  # Store's expected demand in the first step (day)
        self.store_inv = 0              # Store's current inventory 
        self.d_s = 7*[0]                # List of amount of products bought at the Store in the last 7 days (excluding days with zero demand)
               
    def Store_request_restock(self):
        DC = self.model.schedule.agents[N+N_s] # Retrieve the DC agent from the schedule.agents list of the scheduler
        """Calculate the amount to be ordered by the Store"""
        order_data = []
        for d in self.d_s:     # Determining the average daily demand,
            if d > 0:          # exclude days in which the demand is equal to 0
                order_data.append(d)
        order_amount = round(np.mean(order_data)*round(1/f_restock)) # The order amount is equal to the average daily demand multiplied by the restock period [days]
        """The Store sends a restock request to the DC"""
        DC.dc_orders_waitlist = np.concatenate((DC.dc_orders_waitlist,[[self, order_amount]]),axis=0) # The order placed by the Store is added to the list of orders waiting to be processed by the DC. In such list, each order contains the Store agent that placed the order and the requested amount
    
    def step(self):
        
        """In the first step (day), the starting inventory of the Store is determined"""
        if i == 0:
            self.d_s[0:6] = 6*[m.ceil(self.store_expected_demand)] # The expected demand of the Store in the first step (day) is used as an average demand value to initiate the d_s array. The first 6 elements of the d_s array are popuplated with such value to make sure the restock amount can be computed when the actual values of past days' demand are not available yet
            self.store_inv += m.ceil(self.store_expected_demand*(1/f_restock+dc_transport_time)) # The expected demand of the Store in the first step (day), the restock period, the DC-to-store transport time are used to determine the Store's starting inventory
        """Request restock"""
        if ((i+1)*f_restock).is_integer(): # Every restock period, 
            self.Store_request_restock()      # request restock
        """Update the array containing the demand of the past steps (days)""" 
        if self.d_s[-1] != 0:           # If the demand of the step (day), which is equal to the overall quantity purchased at the Store in the step (day), is not 0,
            self.d_s.append(0)          # An additional element of value 0 is added to the array as this element will be updated in the following step (day)
            self.d_s = self.d_s[1:]     # the updated d_s array drops its oldest element 
            
class DistributionCenter(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.dc_expected_demand = 0                         # Distribution Center (DC) expected demand in the first step (day)   
        self.dc_inv = 0                                     # DC's current inventory
        self.dc_orders_waitlist = np.empty([0,2])           # DC's list of pending orders with two dimensions ([Store agent, requested amount])
        self.dc_current_processed_orders = np.empty([0,3])  # DC's list of orders currently being processed with three dimensions ([Store agent, restock amount, remaining DC-to-store transport time])
        self.order = 0                                      # DC's restock amount for the next order to the manufacturer
        self.lost_restock = 0                               # Amount of requested products that the DC is not able to satisfy in the step (day)
    
    def dc_order_processing(self): 
        total_order_q = np.transpose(self.dc_orders_waitlist)[1].sum()        # The requested amounts of all the Store orders in the dc_orders_waitlist are summed
        self.model.orders_at_dc += total_order_q                              # and the resulting quantity stored in the orders_at_dc model's attribute to record the demand received by the DC at each step (day)
        """The pending orders are moved to the dc_current_processed_orders to be processed"""
        
        for n in self.dc_orders_waitlist:
            """Extract the information of each pending order"""
            s_order, q_order = n
            t_order = dc_transport_time
            """Check the DC's remaining inventory and start processing the orders"""
            if self.dc_inv > 0:                                     # If the DC has some stock,
                restock_amount = np.min([q_order,self.dc_inv])      # the DC supplies the Store of the available quantity
                if restock_amount == self.dc_inv:                   # If the DC does not have enough stock to satisfy the entire demand of the Store,
                    self.lost_restock += q_order-restock_amount     # the quantity the DC is not able to satisfy is recorded
                self.dc_inv -= restock_amount                       # The inventory at the DC is updated       
                self.dc_current_processed_orders = np.concatenate((self.dc_current_processed_orders,[[s_order,restock_amount,t_order]]),axis=0) # The order is added to the list of orders currently processed by the DC,
                self.dc_orders_waitlist = np.delete(self.dc_orders_waitlist,0,0)                                                                # and it is removed from the list of pending orders. 
                # Add restocked orders to next restock from manufacturer
                self.order += restock_amount # Add the restock amount to the amount to be ordered to the manufacturer 
            else:                                                   # If the DC has no stock,
                self.lost_restock += q_order                        # the entire quantity requested by the Store is recorded as lost demand
                self.dc_orders_waitlist = np.delete(self.dc_orders_waitlist,0,0) # the other pending orders are dropped completely.
        
        """Process orders"""
        for n in self.dc_current_processed_orders:
            if n[2] > 0:  # If the transport from the DC to the store is not terminated yet, 
                n[2] -= 1 # update transport time.
            else:           # If the restock quantity has arrived to the store, 
                n[0].store_inv += int(n[1]) # update the store inventory
                n[2] -= 1 # and set the DC-to-store transport time to -1 to signal that the order has been processed.
        self.dc_current_processed_orders = np.delete(self.dc_current_processed_orders,np.where(self.dc_current_processed_orders[:,2]==-1),0) # The orders that have been processed are removed from the list. 
   
    def dc_request_restock(self):
        Manufacturer = self.model.schedule.agents[-1]   # Retrieve the Manufacturer agent (i.e. the last one) from the schedule.agents list of the scheduler
        Manufacturer.restock_request = self.order # The restock amount requested by the DC is added to the Manufacturer's restock_requests attribute. 
        
    def step(self):
        self.lost_restock = 0  
        if i ==0:
            self.dc_inv = DC_starting_inventory
        """At each step (day), update pending orders list and process orders"""
        self.dc_order_processing()
        """Request restock""" 
        if ((i+1)*f_restock).is_integer(): # Every restock period, 
            self.dc_request_restock()      # request restock. 
            self.order = 0                 # Reset the restock quantity 
            
class Manufacturer(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.m_capacity = 0                                 # Manufacturer's capacity 
        self.restock_request = 0                            # Restock amount requested by the DC
        self.m_current_processed_orders = np.empty([0,2])   # Manufacturer's list of orders currently being processed ([requested amount, remaining Manufacturer-to-DC transport time])
    
    def m_order_processing(self):    
        if self.restock_request != 0:
            """"Check the Manufacturer's capacity"""
            restock_amount = np.min([self.restock_request, self.m_capacity]) # Determine the portion of the requested quantity that can be satisfied given the Manufacturer's capacity
            """Extract the information for each request"""
            t_order = m_transport_time
            self.m_current_processed_orders = np.concatenate((self.m_current_processed_orders,[[restock_amount,t_order]]),axis=0) # The order, containing the actual quantity that the Manufacturer can restock and the Manufacturer-to-DC transport time, is added to the list of orders currently processed by the Manufacturer,
            self.restock_request = 0 # and the restock amount requested by the DC is reset
        """Process orders"""
        for n in self.m_current_processed_orders:
            if n[1] > 0:    # If the transport from the Manufacturer to the DC is not terminated yet,
                n[1] -= 1   # update transport time.
            else:           # If the restock quantity has arrived to the DC,
                dc = self.model.schedule.agents[N+N_s] # retrieve the DC agent from the schedule.agents list of the scheduler,
                dc.dc_inv += int(n[0]) # update the DC inventory,
                n[1] -= 1              # and set the DC-to-store transport time to -1 to signal that the order has been processed.
        self.m_current_processed_orders = np.delete(self.m_current_processed_orders,np.where(self.m_current_processed_orders[:,1]==-1),0) # The orders that have been processed are removed from the list 
   
    def step(self):
        """In the first step (day), the capacity of the Manufacturer is defined"""
        if i == 0:
            self.m_capacity = m_starting_capacity
        """At each step (day), update pending orders list and process orders"""
        self.m_order_processing()

"""Model class definition"""        
class ABSModel(Model):
    def __init__(self, t_run):
        self.schedule = BaseScheduler(self)         # Activates all agents once per step, in order of addition to schedule
        self.d = 0                                  # Demand of current step (day)
        self.pandemic_count = 0                     # Counts the amount of steps (days) from the pandemic start
        self.grid = SingleGrid(w_grid,h_grid,True)  # Grid on which Customer agents and Store agents are positioned
        self.initial_panic_customers = []           # List of Customers in panic in the step (day)
        self.orders_at_dc = 0                       # Records the demand received by the DC in the step (day)
        self.customer_list_region=[]                # List of customer agents per region
        print("Starting agents creation...")
        
        """Agents' and grid initialization"""
        customer_list = [*range(N)]
        store_list = [*range(N,N+N_s)]
        w_count = [*range(int(w_grid/wh_region))] 
        h_count = [*range(int(h_grid/wh_region))] 
        available_grid_list = [*range(w_grid*h_grid)]
        available_grid_list_stores = []

        # Create map:
        self.responsiveGraph = ResponsiveGraph(t_run)
    
        """Customer creation and placement on the grid"""    
        for h in h_count:    
            for w in w_count:
                customer_list_region = []
                i_region = w+len(w_count)*h
                available_region_list = available_grid_list[0:wh_region*wh_region]
                available_consumption_list = consumption_list_region[:]
                # Place a set amount of customers on a region
                while len(customer_list) > len([*range(N)])/N_regions*(N_regions-1-i_region): 
                    i_customer = random.choice(customer_list)
                    a = Customer(i_customer, self)  
                    # Add the customer agent to the scheduler
                    self.schedule.add(a) 
                    customer_list_region=np.append(customer_list_region, [a])
                    customer_list.remove(i_customer)
                    # Place the Customer agent on a unoccupied grid location
                    location_customer = random.choice(available_region_list) 
                    location_customer_region = location_customer-i_region*wh_region*wh_region 
                    y_region = int(np.floor(location_customer_region/wh_region))
                    x_region = location_customer_region-y_region*wh_region
                    y = y_region+ wh_region*h 
                    x = wh_region*w+x_region
                    self.grid.place_agent(a,(x,y))
                    available_region_list.remove(location_customer)
                    # Assign consumption behavior
                    consumption_customer = random.choice(available_consumption_list)
                    a.f_base = consumption_customer[0]
                    a.f = consumption_customer[0]
                    a.q_base = consumption_customer[1]
                    a.q = consumption_customer[1]
                    available_consumption_list.remove(consumption_customer)
                # Update available grid locations
                available_grid_list = available_grid_list[wh_region*wh_region:]
                available_grid_list_stores = np.append(available_grid_list_stores, available_region_list)
                available_grid_list_stores=[int(i) for i in available_grid_list_stores]
            
        """Stores creation and placement on the grid""" 
        for h in h_count:
            for w in w_count:
                i_region = w+len(w_count)*h
                available_region_list = available_grid_list_stores[0:wh_region*wh_region-int(N/N_regions)]
                while len(store_list) > len([*range(N_s)])/N_regions*(N_regions-1-i_region): 
                    i_store = random.choice(store_list)
                    b = Store(i_store, self)
                    # Add the Store agent to the scheduler after customers
                    self.schedule.add(b)
                    store_list.remove(i_store)
                    # Place the Store agent on a unoccupied grid location
                    location_store = random.choice(available_region_list)
                    location_store_region = location_store-i_region*wh_region*wh_region 
                    y_region = int(np.floor(location_store_region/wh_region))
                    x_region = location_store_region-y_region*wh_region
                    y = y_region+wh_region*h 
                    x = wh_region*w+x_region
                    self.grid.place_agent(b, (x,y))
                    store_positions.append(b.pos)
                    available_region_list.remove(location_store)
                available_grid_list_stores = available_grid_list_stores[wh_region*wh_region-int(N/N_regions):]
        
        """DC creation""" 
        dc_list = [*range(N+N_s,N+N_s+N_dc)]
        while len(dc_list) > 0:
            i_dc = random.choice(dc_list)
            c = DistributionCenter(i_dc, self)
            # Add the DC agent to the scheduler
            self.schedule.add(c)
            dc_list.remove(i_dc)
            
        """Manufacurer creation"""
        m_list = [*range(N+N_s+N_dc,N+N_s+N_dc+N_m)]
        while len(m_list) > 0:
            i_m = random.choice(m_list)
            d = Manufacturer(i_m, self)
            # Add the Manufacturer agent to the scheduler
            self.schedule.add(d)
            m_list.remove(i_m)
        print("All agents have been scheduled")
        
    def creating_panic_customers(self):
        pbs_customer_list = []               # List of Customer agents prone to panic buying 
        alpha_pbs = random.uniform(0.02, 0.04) # Percentage of customers prone to panic buying
        N_pb = m.floor(alpha_pbs*N)          # Amount of customers prone to panic buying
        self.initial_panic_customers=[]      # List of Customer agents in panic in the step (day)
        
        """For each region, define a certain amount of Customer agents that are in panic"""
        for w in (range(N_regions)):
            self.customer_list_region=self.schedule.agents[w*int(N/N_regions):((w+1)*int(N/N_regions))]
            while len(pbs_customer_list) < (N_pb*(w+1))/N_regions: 
                a = random.choice( self.customer_list_region)
                pbs_customer_list = np.append(pbs_customer_list,[a])
                if len(self.initial_panic_customers) < (N_pb*(w+1))/N_regions: 
                    self.initial_panic_customers = np.append(self.initial_panic_customers,[a]) 
                self.customer_list_region.remove(a)
        
    def step(self):
        """If the pandemic has started"""
        if pandemic == True:
            self.pandemic_count += 1 # the count of the pandemic days is updated
            
            """Until panic end, at each step(day) a percentage of Customer agents is in panic""" 
            if i<panic_end: 
                self.creating_panic_customers()
                print("Day", i, ": panic behavior")
            else: 
                print("Day", i, ": back to regular behavior")
        else:
            self.pandemic_count = 0
            print("Day", i, ": regular behavior")
            
        """Advances the simulation time of 1 step (day)"""
        self.schedule.step()
        
        for y in range(h_grid):
            for x in range(w_grid):
                cellmates = self.grid.get_cell_list_contents((x,y))
                if len(cellmates) > 0:
                    cus = self.random.choice(cellmates)
                    if isinstance(cus, Customer):
                        if cus.hungry:
                            self.responsiveGraph.agents_in_cell[y][x] = -3
                        elif cus.pbs:
                            self.responsiveGraph.agents_in_cell[y][x] = -2
                        else:
                            self.responsiveGraph.agents_in_cell[y][x] = -1
        lr = 0
        for a in self.schedule.agents:
            if hasattr(a,"dc_inv"):
                lr += a.lost_restock
        ifr = 100
        if self.orders_at_dc != 0:
            ifr = (self.orders_at_dc-lr)/self.orders_at_dc*100
        self.responsiveGraph.update(i, self.d, ifr)


       
"""Initializing model parameters"""
n_runs = 1                              # number of runs
t_run =  50 #100                            # number of steps (days) in a run
pandemic_start = 15                     # Start of pandemic 
panic_end = pandemic_start+20           # Moment in which tha panic behavior of Customers terminates
pandemic = False                        # It becomes True when pandemic is active

"""Initializing arrays"""
customer_positions=[]
store_positions=[]
demand = np.zeros([t_run,n_runs])
orders_at_dc = np.zeros([t_run,n_runs])
lost_dc_inv = np.zeros([t_run,n_runs,N_dc])

"""Running the simulation"""
for r in range(n_runs):
    print("\n\nRun number:" + str(r))
    model_1 = ABSModel(t_run)
    pandemic = False
    model_1.pandemic_count = 0
    
    """Run for t_run steps"""
    for i in range(t_run):
        # Pandemic check
        if i >= pandemic_start:
            pandemic = True
        else:
            pandemic = False
        """ Model step """
        model_1.step()
        """Extract data from the model for further analysis"""
        demand[i][r] = model_1.d
        model_1.d = 0
        orders_at_dc[i][r] = model_1.orders_at_dc
        model_1.orders_at_dc = 0
        
        for a in model_1.schedule.agents:
            if hasattr(a,"dc_inv"):
                lost_dc_inv[i][r][a.unique_id-N-N_s] = a.lost_restock



""""Averages over runs"""
av_demand = demand.mean(1)
av_orders_at_dc = orders_at_dc.mean(1)
av_lost_dc_inv = lost_dc_inv.mean(1).sum(1)

"""Performance indicators"""
# 1. Initialize array
ifr_dc = np.zeros(t_run)

# 2.Compute the item fill rate of the DC per step
for i in range(t_run): 
    if av_orders_at_dc[i] == 0:
        ifr_dc[i] = 1
    else:
        ifr_dc[i] = (av_orders_at_dc[i]-av_lost_dc_inv[i])/av_orders_at_dc[i]

print("Minimal Distribution Center IFR = ", round(np.min(ifr_dc)*100, 2), "%\n\n")