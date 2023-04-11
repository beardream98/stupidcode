#实现item 和len 两个特殊方法
import collections

#namedtuple 创建只有少数属性但是没有方法的对象
Card=collections.namedtuple("Card",["rank","suit"])

class FrenchDeck:
    #属于类的变量，可以通过FrenchDeck.ranks 访问
    # __双下划綫代表私有，只允许类访问，_单下划线代表保护，只允许类和子类访问
    ranks=[str(n) for n in range(2,11)]+list("JQKA")
    suits="spades diamonds clubs headrts".split()
    suit_values=dict(spades=3,headrts=2,diamonds=1,clubs=0)

    def __init__(self) -> None:
        self._cards=[Card(rank,suit)for suit in self.suits for rank in self.ranks]
    def __len__(self):
        return len(self._cards)
    def __getitem__(self,index):

        return self._cards[index]
    def spades_high(self,card):
        rank_value=FrenchDeck.ranks.index(card.rank)
        return rank_value*len(self.suit_values)+self.suit_values[card.suit]



deck=FrenchDeck()
from random import choice
choice(deck)

#m::n 操作可以从m开始没隔n个选取一次元素
# deck[12::13]
for card in sorted(deck,key=deck.spades_high):
    print(card)


if __name__ == '__main__':
    print(__name__)
