
#ifndef MYLIST_H
#define MYLIST_H

// Note: There is never an implementation file (*.C) for a template class

template <class T>
class MyList
{

public:
  MyList *next;
  T *data;

public:
  MyList();
  MyList(T *p);
  ~MyList();
  void insert(T *p);
  void clearList();
  void destroyList();
  void catList(MyList<T> *p);
  void CloneList(MyList<T> *p);
};

template <class T>
MyList<T>::MyList()
{
  data = 0;
  next = 0;
}
template <class T>
MyList<T>::MyList(T *p)
{
  data = p;
  next = 0;
}

template <class T>
MyList<T>::~MyList()
{
}
template <class T>
void MyList<T>::insert(T *p)
{
  MyList *ct = this;
  if (data == 0)
  {
    data = p;
  }
  else
  {
    while (ct->next)
    {
      ct = ct->next;
    }
    ct->next = new MyList(p);
    ct = ct->next;
    ct->next = 0;
  }
}
template <class T>
void MyList<T>::clearList()
{
  MyList *ct = this, *n;
  while (ct)
  {
    n = ct->next;
    delete ct;
    ct = n;
  }
}
template <class T>
void MyList<T>::destroyList()
{
  MyList *ct = this, *n;
  while (ct)
  {
    n = ct->next;
    delete ct->data;
    delete ct;
    ct = n;
  }
}
template <class T>
void MyList<T>::catList(MyList<T> *p)
{
  MyList *ct = this;
  while (ct->next)
  {
    ct = ct->next;
  }
  ct->next = p;
}
template <class T>
void MyList<T>::CloneList(MyList<T> *p)
{
  MyList *ct = this;
  p = 0;
  while (ct)
  {
    if (!p)
      p = new MyList<T>(ct->data);
    else
      p->insert(ct->data);
    ct = ct->next;
  }
}
#endif /* MyList_H */
