import heapq
import pandas as pd
from typing import List


class Node:
    """
    A generic class implementing the common functionality of nodes and switches.
    The node_id is used to represent the node.

    The routes is a dictionary that is used to forward a packet.
    The destination is stored as the key in the dictionary and the value is the next hop.
    Given a packet, you can retrieve the next by looking for its destination in routes.

    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.routes = {}

    def add_route(self, destination, next_hop):
        self.routes[destination] = next_hop

    def next_hop(self, destination):
        if self == destination:
            return self
        return self.routes[destination]


class Packet:
    """
    Represent a packet. Currently, a packet includes only the source and destination, which are
    usually included in the header of the packet. A unique packet id is given to each packet
    upon its creating.
    """
    cnt: int = 0

    def __init__(self, source: Node, destination: Node):
        self.packet_id = Packet.cnt
        Packet.cnt += 1
        self.source = source
        self.destination = destination

    def __str__(self):
        return f'P[{self.packet_id},{self.source.node_id}->{self.destination.node_id}]'


class Host(Node):
    """
    A host includes a FIFO queue that stores the packets to be transmitted
    """

    def __init__(self, node_id):
        super().__init__(node_id)
        self.output_queue = []

    def __str__(self):
        return f'{self.node_id:2s} queue={[p.packet_id for p in self.output_queue]}'


class Switch(Node):
    """
    The class emulate the behavior of a switch/router.
    Note that unlike a host, the switch has both an input and an output queue.
    """

    def __init__(self, node_id, processing_delay=0):
        super().__init__(node_id)
        self.input_queue: List[Packet] = []
        self.output_queue: List[Packet] = []
        self.processing_delay = processing_delay

    def __str__(self):
        return f'{self.node_id:2s} in={[p.packet_id for p in self.input_queue]} out={[p.packet_id for p in self.output_queue]}'


class Event:
    """
    This class holds the information about the events that will be interpreted by the
    simulator. The event has the following state
    - target_node - the node that needs to handle the packet
    - event_type - it can be either ENQUEUE, TRANSMIT, PROPAGATE, RECEIVE
    - time - the time when the event will be executed
    - packet - the packet associated with the event (can be None)
    - event_id - the id of the event

    """
    ENQUEUE = 0
    TRANSMIT = 1
    PROPAGATE = 2
    RECEIVE = 3

    cnt = 0

    def __init__(self, event_type: int, target_node: Node, packet: Packet = None, time: int = None):
        assert (0 <= event_type <= 3)
        self.target_node = target_node
        self.event_type = event_type
        self.time = time
        self.packet = packet
        self.event_id = Event.cnt
        Event.cnt += 1

    def type_to_str(self):
        if self.event_type == Event.ENQUEUE:
            return 'ENQUEUE'
        elif self.event_type == Event.TRANSMIT:
            return 'TRANSMIT'
        elif self.event_type == Event.PROPAGATE:
            return 'PROPAGATE'
        elif self.event_type == Event.RECEIVE:
            return 'RECEIVE'
        else:
            raise Exception('Unknown event type')

    def __str__(self):
        return f'{self.time:4d} {self.type_to_str():12s} {self.target_node.node_id} pkt={str(self.packet)}'


class Simulator:
    """
    The main simulator class.
    """

    def __init__(self, transmission_delay=10, propagation_delay=1):
        self.event_queue: List[Event] = []
        self.transmission_delay: int = transmission_delay
        self.propagation_delay: int = propagation_delay
        self.clock = 0
        self.nodes = {}
        self.link_data = pd.DataFrame({'Seq num.': [], 'Queue @A': [], 'Transmit @A': [], 'Propagate @A': [], 'Receive @B': []})

    def schedule_event_after(self, event: Event, delay: int):
        """
        Schedules an event to be executed in the future

        :param event:
        :param delay: - the delay after which the event will be executed
        :return:
        """
        event.time = self.clock + delay
        heapq.heappush(self.event_queue, (event.time, event.cnt, event))

    def run(self):
        """
        Runs the simulator.

        :return:
        """
        print('Starting simulation')
        # self.print_event_queue()
        while len(self.event_queue) > 0:
            self.clock, _, event = heapq.heappop(self.event_queue)
            # print("Time: " + str(self.clock))
            # print(f'{str(event)}')
            self.handle_event(event)
            self.log_link_data(event) #log data for link experiment
            self.link_data.to_csv('single_link.csv', index=False)
        print('Simulation ended')

    def handle_event(self, event):
        """
        Handles the execution of the events. You must implement this

        :param event:
        :return:
        """

        # YOU NEED TO IMPLEMENT THIS METHOD
        
        if(event.event_type == Event.ENQUEUE):
            # print("ENQUEUE to " + str(event.packet.source.node_id) + " at time " + str(self.clock))
            self.schedule_event_after(Event(Event.TRANSMIT, event.packet.source, event.packet, self.clock + len(event.packet.source.output_queue) * self.transmission_delay), 
                                        len(event.packet.source.output_queue)*self.transmission_delay) #schedule transmit event with respect to queueing delay
            event.target_node.output_queue.append(event.packet) #add packet to output queue of source
        
        elif(event.event_type == Event.TRANSMIT):
            # print("TRANSMIT packet " + str(event.packet.packet_id) + " at " + str(event.packet.source.node_id) + " at time " + str(event.time))
            self.schedule_event_after(Event(Event.PROPAGATE, event.packet.source, event.packet, event.time + self.transmission_delay), self.transmission_delay) #schedule propagate event
            self.clock += self.transmission_delay #increment clock by transmission delay
        
        elif(event.event_type == Event.PROPAGATE):
            # print("PROPAGATE packet " + str(event.packet.packet_id) + " from " + str(event.packet.source.node_id) + " to " + str(event.packet.destination.node_id) + " at time " + str(event.time))
            self.schedule_event_after(Event(Event.RECEIVE, event.packet.destination, event.packet, event.time + self.propagation_delay), 
                                        self.propagation_delay) #schedule receive event with repect to propagation delay
            event.packet.source.output_queue.remove(event.packet) #remove packet from output queue of source
            self.clock += self.propagation_delay #increment clock by propagation delay

        elif(event.event_type == Event.RECEIVE):
            # print("Node " + str(event.packet.destination.node_id) + " RECIEVEs packet " + str(event.packet.packet_id) + " at time " + str(event.time))
            pass

        # self.print_event_queue()
        # print("A: " + str(self.nodes['A'].output_queue))
        # print("B: " + str(self.nodes['B'].output_queue))
        # print("--------------------")

    #logs the data into the dataframe for the link experiment
    def log_link_data(self, event : Event):
        match event.event_type:
            case Event.ENQUEUE:
                self.link_data.loc[len(self.link_data)] = {'Seq num.': event.packet.packet_id, 'Queue @A': event.time, 'Transmit @A': None, 'Propagate @A': None, 'Receive @B': None}
            case Event.TRANSMIT:
                self.link_data.loc[event.packet.packet_id, 'Transmit @A'] = event.time
            case Event.PROPAGATE:
                self.link_data.loc[event.packet.packet_id, 'Propagate @A'] = event.time
            case Event.RECEIVE:
                self.link_data.loc[event.packet.packet_id, 'Receive @B'] = event.time

    #prints the event queue
    def print_event_queue(self):
        print("event_queue: ", end='')
        for _, _, event in self.event_queue:
            print(f'{str(event.event_type)}' + f'{str(event.packet.source.node_id)}' + f'{str(event.time)}', end=', ')
        print()
        

    def new_node(self, node_id: str):
        if node_id in self.nodes:
            raise Exception('Node already added')
        node = Host(node_id)
        self.nodes[node_id] = node
        return node

    def new_switch(self, str_id, processing_delay):
        if str_id in self.nodes:
            raise Exception('Node already added')
        switch = Switch(str_id, processing_delay=processing_delay)
        self.nodes[str_id] = switch
        return switch


def link_experiment():
    sim = Simulator()
    A, B = sim.new_node('A'), sim.new_node('B')
    A.add_route(B, B)

    sim.schedule_event_after(Event(Event.ENQUEUE, A, Packet(A, B)), 0)
    sim.schedule_event_after(Event(Event.ENQUEUE, A, Packet(A, B)), 0)
    sim.run()


def switch_experiment():
    sim = Simulator()
    A, B, C, D = sim.new_node('A'), sim.new_node('B'), sim.new_switch('C', processing_delay=1), sim.new_node('D')
    A.add_route(D, C)
    B.add_route(D, C)
    C.add_route(D, D)

    "You will have to setup the workload for this experiment"
    sim.run()


if __name__ == '__main__':
    link_experiment()  # switch_experiment()
