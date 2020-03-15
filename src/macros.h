//
// Created by Hetul on 08/02/20.
//

#ifndef CUSTOM_CTC_MACROS_H
#define CUSTOM_CTC_MACROS_H

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

#endif //CUSTOM_CTC_MACROS_H
