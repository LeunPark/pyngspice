#include <stdio.h>
#include <string.h>

#include <Python.h>
#include <structmember.h>
#include "sharedspice.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define DEFAULT_ARRAY_CAPACITY 50

// Asterisk (*) indicates that the variable is used internally.

static PyObject *instances_dict = NULL;  // *
static PyObject *NgSpiceCommandError;

typedef struct {
    char **data;
    size_t size;
    size_t capacity;
} string_array_t;

typedef struct {
    PyObject_HEAD
    int ngspice_id;

    string_array_t stdout_;
    string_array_t stderr_;

    bool error_in_stdout;  // *
    bool error_in_stderr;  // *

    bool is_running;

    bool has_send_char;  // *
} shared_t;

// ====================================================================================
// Helper Functions

static bool error_check(const char *message) {
//    const char *end = message + strlen(message);
//    while (message + 4 < end) {
//        // TODO: A hacky way to check just "rror" or "RROR" after 'e' char, considering the frequency..
//        if (*message == 'e' && !memcmp(message, "error", 5))
//            return true;
//        else if (*message == 'E' && (!memcmp(message, "Error", 5) || !memcmp(message, "ERROR", 5)))
//            return true;
//        message++;
//    }
    const char *end = message + strlen(message) - 4;
    for (const char *cur = message; cur < end; cur++) {
        if ((*cur == 'e' || *cur == 'E') &&
            (cur[1] == 'r' || cur[1] == 'R') &&
            (cur[2] == 'r' || cur[2] == 'R') &&
            (cur[3] == 'o' || cur[3] == 'O') &&
            (cur[4] == 'r' || cur[4] == 'R'))
            return true;
    }
    return false;
}

static inline int handle_callback(char *method_name, PyObject *res_obj) {
    if (res_obj == NULL) {
        PyErr_Format(PyExc_RuntimeError, "User method %s failed.", method_name);
        return -1;
    }

    long result = PyLong_AsLong(res_obj);
    if (result == -1 && PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "Expected %s to return an int, got %s instead.", method_name, Py_TYPE(res_obj)->tp_name);
        return -1;
    }
    Py_DECREF(res_obj);

    return (int)result;
}

static bool check_method(shared_t *self, char *method_name) {
    if (PyObject_HasAttrString((PyObject *)self, method_name) == 0)
        return false;

    PyObject *method = PyObject_GetAttrString((PyObject *)self, method_name);
    if (method == NULL) {
        PyErr_Format(PyExc_AttributeError, "Failed to get method %s.", method_name);
        return false;
    }

    bool is_callable = PyCallable_Check(method);
    Py_DECREF(method);

    return is_callable;
}

static int string_array_init(string_array_t *array) {
    array->size = 0;
    array->capacity = DEFAULT_ARRAY_CAPACITY;
    array->data = malloc(DEFAULT_ARRAY_CAPACITY * sizeof(char *));
    if (!array->data) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

static int string_array_append(string_array_t *array, const char *value) {
    if (array->size >= array->capacity) {
        array->capacity *= 2;
        char **new_data = realloc(array->data, array->capacity * sizeof(char *));
        if (!new_data) {
            PyErr_NoMemory();
            return -1;
        }
        array->data = new_data;
    }
//    (*array)[(*size)++] = strdup(value);
    array->data[array->size++] = strdup(value);
    return 0;
}

static inline void _string_array_free(string_array_t *array) {
    for (int i = 0; i < array->size; i++)
        free(array->data[i]);
    free(array->data);
}

static int string_array_clear(string_array_t *array) {
    _string_array_free(array);
    array->size = 0;

    char **new_data = malloc(DEFAULT_ARRAY_CAPACITY * sizeof(char *));
    if (!new_data) {
        PyErr_NoMemory();
        return -1;
    }
    array->data = new_data;
    array->capacity = DEFAULT_ARRAY_CAPACITY;
    return 0;
}

static inline PyObject *join_string_array(string_array_t *array) {
    if (array->size == 0)
        return PyUnicode_New(0, 127);

    const char sep = '\n';
    size_t total_len = 0;
#ifdef _MSC_VER
    size_t *string_lens = malloc(array->size * sizeof(size_t));
#else
    size_t string_lens[array->size];
#endif
    for (int i = 0; i < array->size; i++) {
        total_len += (string_lens[i] = strlen(array->data[i])) + 1;
    }

    // PyUnicode_New: New in version 3.3
    PyObject *result = PyUnicode_New(total_len - 1, 255);  // 255: might encounter unicode error?
    if (!result) {
        PyErr_NoMemory();
        return NULL;
    }

    Py_UCS1 *p = PyUnicode_1BYTE_DATA(result);
    char *pos = (char *)p;
//    char *pos = result;
    for (int i = 0; i < array->size; i++) {
        memcpy(pos, array->data[i], string_lens[i]);
        pos += string_lens[i];
        if (i < array->size - 1)
            *pos++ = sep;
    }
    *pos = '\0';
#ifdef _MSC_VER
    free(string_lens);
#endif

    return result;
}

static inline PyObject *string_array_to_list(string_array_t *array) {
    PyObject *list = PyList_New(array->size);
    if (!list)
        return NULL;

    for (int i = 0; i < array->size; i++) {
        PyObject *str = PyUnicode_FromString(array->data[i]);
        if (!str) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, str);
    }
    return list;
}

// ====================================================================================
// ngSpice Shared Callbacks

static int send_char_callback(char *message, int ngspice_id, void *user_data) {
    shared_t *self = (shared_t *)user_data;

    char *delimiter_pos = strchr(message, ' ');
    if (delimiter_pos == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid message format of send_char_callback message.");
        return -1;
    }

    size_t prefix_len = delimiter_pos - message;
    char *content = delimiter_pos + 1;

    // Handling standard stream messages
    // Note that there is an exceptional message such that " Reference value :  0.00000e+00"
    if (prefix_len == 6 && !memcmp(message, "stderr", 6)) {
        if (string_array_append(&self->stderr_, content) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to append to stderr.");
            return -1;
        }

        if (strncmp(content, "Note:", 5) == 0 ||
            strncmp(content, "Warning:", 8) == 0 ||
            strncmp(content, "Using", 5) == 0) {
            // TODO: LOGGER WARNING
        } else {
            self->error_in_stderr = true;
        }
        // temporary measure to print in red
        PySys_WriteStderr("\033[1;31m%s\033[0m\n", content);
    } else {
        if (string_array_append(&self->stdout_, content) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to append to stderr.");
            return -1;
        }

        if (error_check(content))
            self->error_in_stdout = true;
    }

    if (self->has_send_char) {
        PyObject *res_obj = PyObject_CallMethod((PyObject *)self, "send_char", "si", message, ngspice_id);
        return handle_callback("send_char", res_obj);
    }
    return 0;
}

static int send_stat_callback(char *message, int ngspice_id, void *user_data) {
    shared_t *self = (shared_t *)user_data;

    PyObject *res_obj = PyObject_CallMethod((PyObject *)self, "send_stat", "si", message, ngspice_id);
    return handle_callback("send_stat", res_obj);
}

static int exit_callback(int exit_status, bool immediate_unloading, bool quit_exit, int ngspice_id, void *user_data) {
    // TODO: Implement this

//    printf("exit_callback %d %d %d %d\n", exit_status, immediate_unloading, quit_exit, ngspice_id);
    return exit_status;
}

static int send_data_callback(pvecvaluesall data, int number_of_vectors, int ngspice_id, void *user_data) {
    shared_t *self = (shared_t *)user_data;

    PyObject *vectors = PyDict_New();
    if (!vectors) return -1;

    for (int i = 0; i < number_of_vectors; i++) {
        pvecvalues vec = data->vecsa[i];
        PyObject *vector_name = PyUnicode_FromString(vec->name);
        if (!vector_name) goto cleanup;

        PyObject *value = PyComplex_FromDoubles(vec->creal, vec->cimag);
        if (!value) {
            Py_DECREF(vector_name);
            goto cleanup;
        }

        if (PyDict_SetItem(vectors, vector_name, value) < 0) {
            Py_DECREF(vector_name);
            Py_DECREF(value);
            goto cleanup;
        }

        Py_DECREF(vector_name);
        Py_DECREF(value);
    }

    PyObject *res_obj = PyObject_CallMethod((PyObject *)self, "send_data", "Oii", vectors, number_of_vectors, ngspice_id);
    Py_DECREF(vectors);
    return handle_callback("send_data", res_obj);

cleanup:
    Py_DECREF(vectors);
    return -1;
}

static int send_init_data_callback(pvecinfoall data, int ngspice_id, void *user_data) {
    // TODO: Implement this
//    printf(">>> send_init_data\n");
//    int number_of_vectors = data->veccount;
//
//    for (int i = 0; i < number_of_vectors; i++) {
//        printf("  Vector: %s\n", data->vecs[i]->vecname);
//    }

    return 0;
}

bool no_bg = true;
static int bg_thread_running_callback(bool noruns, int ngspice_id, void *user_data) {
    // TODO: Implement this

    no_bg = noruns;
    if (noruns)
        printf("bg not running\n");
    else
        printf("bg running\n");

    return 0;
}

static int get_vsrc_data_callback(double *voltage, double time, char *node_name, int ngspice_id, void *user_data) {
    shared_t *self = (shared_t *)user_data;

    PyObject *res_obj = PyObject_CallMethod((PyObject *)self, "get_vsrc_data", "ddsi", *voltage, time, node_name, ngspice_id);
    return handle_callback("get_vsrc_data", res_obj);
}

static int get_isrc_data_callback(double *current, double time, char *node_name, int ngspice_id, void *user_data) {
    shared_t *self = (shared_t *)user_data;

    PyObject *res_obj = PyObject_CallMethod((PyObject *)self, "get_isrc_data", "ddsi", *current, time, node_name, ngspice_id);
    return handle_callback("get_isrc_data", res_obj);
}

static int get_sync_data_callback(double actual_time, double *delta_time, double old_delta_time, int redostep, int ngspice_id, int loc, void *user_data) {
    shared_t *self = (shared_t *)user_data;

    PyObject *res_obj = PyObject_CallMethod((PyObject *)self, "get_sync_data", "dddiii", actual_time, *delta_time, old_delta_time, redostep, ngspice_id, loc);
    return handle_callback("get_sync_data", res_obj);
}

// ====================================================================================
// Shared Methods

static PyObject *shared_new_instance(PyObject *cls, PyObject *args, PyObject *kwds) {
    int ngspice_id = 0;
    PyObject *send_data = Py_False, *verbose = Py_False;

    static char *kwlist[] = {"ngspice_id", "send_data", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iOO", kwlist, &ngspice_id, &send_data, &verbose))
        return NULL;

    if (instances_dict == NULL) {
        instances_dict = PyDict_New();
        if (instances_dict == NULL)
            return NULL;
    }

    PyObject *instance_key = PyLong_FromLong(ngspice_id);
    PyObject *instance = PyDict_GetItemWithError(instances_dict, instance_key);
    if (instance != NULL) {
        Py_INCREF(instance);
    } else {
        instance = PyObject_CallFunctionObjArgs(cls, instance_key, send_data, verbose, NULL);
        if (instance == NULL) {
            Py_DECREF(instance_key);
            return NULL;
        }
        if (PyDict_SetItem(instances_dict, instance_key, instance)) {
            Py_DECREF(instance);
            Py_DECREF(instance_key);
            return NULL;
        }
    }

    Py_DECREF(instance_key);
    return instance;
}

static int shared___init__(shared_t *self, PyObject *args, PyObject *kwds) {
    int ngspice_id = 0;
    PyObject *send_data = Py_False, *verbose = Py_False;

    static char *kwlist[] = {"ngspice_id", "send_data", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iOO", kwlist, &ngspice_id, &send_data, &verbose))
        return -1;

    self->ngspice_id = ngspice_id;

    string_array_init(&self->stdout_);
    string_array_init(&self->stderr_);

//    self->stdout = PyList_New(0);
//    self->stderr = PyList_New(0);
    self->error_in_stderr = false;
    self->error_in_stdout = false;

    self->is_running = false;

//    self->is_inherited = Py_TYPE(self)->tp_base != &PyBaseObject_Type;
    self->has_send_char = check_method(self, "send_char");

    Py_XINCREF(send_data);
    PyObject *res_obj = PyObject_CallMethod((PyObject *)self, "_init_ngspice", "O", send_data);
    Py_XDECREF(send_data);

    if (res_obj == NULL)
        return -1;
    Py_DECREF(res_obj);

    return 0;
}

static PyObject *shared__init_ngspice(shared_t *self, PyObject *args, PyObject *kwds) {
    PyObject *send_data = Py_False;
    int rc;

    static char *kwlist[] = {"send_data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &send_data))
        return NULL;

    void *_send_stat_c = check_method(self, "send_stat") ? send_stat_callback : NULL;
    void *_send_data_c = send_data == Py_True && check_method(self, "send_data") ? send_data_callback : NULL;
    void *_send_init_data_c = check_method(self, "send_init_data") ? send_init_data_callback : NULL;

    rc = ngSpice_Init(send_char_callback, _send_stat_c, NULL, _send_data_c, _send_init_data_c, bg_thread_running_callback, self);
    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "ngSpice_Init failed; got %d.", rc);
        return NULL;
    }

    void *_get_vsrc_data_c = check_method(self, "get_vsrc_data") ? get_vsrc_data_callback : NULL;
    void *_get_isrc_data_c = check_method(self, "get_isrc_data") ? get_isrc_data_callback : NULL;
    rc = ngSpice_Init_Sync(_get_vsrc_data_c, _get_isrc_data_c, NULL, &self->ngspice_id, self);
    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "ngSpice_Init_Sync failed; got %d.", rc);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *shared_clear_output(shared_t *self) {
//    PyList_SetSlice(self->stdout, 0, PyList_Size(self->stdout), NULL);
    string_array_clear(&self->stdout_);
//    PyList_SetSlice(self->stderr, 0, PyList_Size(self->stdout), NULL);
    string_array_clear(&self->stderr_);
    self->error_in_stdout = false;
    self->error_in_stderr = false;

    Py_RETURN_NONE;
}

static PyObject *shared__stdout_getter(shared_t *self, void *closure) {
//    Py_INCREF(self->stdout);
//    return self->stdout;
    return string_array_to_list(&self->stdout_);
}

static PyObject *shared__stderr_getter(shared_t *self, void *closure) {
//    Py_INCREF(self->stderr);
//    return self->stderr;
    return string_array_to_list(&self->stderr_);
}

static PyObject *shared_stdout_getter(shared_t *self, void *closure) {
//    return join_string(self->stdout);
    return join_string_array(&self->stdout_);
}

static PyObject *shared_stderr_getter(shared_t *self, void *closure) {
//    return join_string(self->stdout);
    return join_string_array(&self->stderr_);
}

static PyObject *shared_exec_command(shared_t *self, PyObject *args, PyObject *kwds) {
    char *command;
    PyObject *join_lines = Py_True;

    static char *kwlist[] = {"command", "join_lines", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|O", kwlist, &command, &join_lines))
        return NULL;

    PyObject *res_obj = shared_clear_output(self);
    if (res_obj == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "clear_output failed");
        return NULL;
    }
    Py_DECREF(res_obj);

    int rc = ngSpice_Command(command);
    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "ngSpice_Command('%s') failed; got %d.", command, rc);
        return NULL;
    }

    if (self->error_in_stdout || self->error_in_stderr) {
        PyErr_Format(NgSpiceCommandError, "Command '%s' failed.", command);
        return NULL;
    }

    if (join_lines == Py_True) {
        return shared_stdout_getter(self, NULL);
    } else {
        return shared__stdout_getter(self, NULL);
    }
}

static PyObject *shared_load_circuit(shared_t *self, PyObject *args) {
    char *circuit;
    if (!PyArg_ParseTuple(args, "s", &circuit))
        return NULL;

    size_t capacity = 50;
    char **lines = malloc(capacity * sizeof(char *));
    char *circuit_copy = strdup(circuit);
    if (!lines || !circuit_copy) {
        PyErr_NoMemory();
        goto cleanup;
    }

    int count = 0;
    char *line = strtok(circuit_copy, "\n");
    while (line) {
        if (count >= capacity) {
            capacity *= 2;
            char **new_lines = realloc(lines, capacity * sizeof(char *));
            if (!new_lines) {
                PyErr_NoMemory();
                goto cleanup;
            }
            lines = new_lines;
        }
        lines[count++] = line;
        line = strtok(NULL, "\n");
    }
    lines[count] = NULL;

    PyObject *res_obj = shared_clear_output(self);
    if (res_obj == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "clear_output failed");
        goto cleanup;
    }
    Py_DECREF(res_obj);

    int rc = ngSpice_Circ(lines);
    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "ngSpice_Circ failed; got %d.", rc);
        goto cleanup;
    }

    if (self->error_in_stdout || self->error_in_stderr) {
        PyErr_SetString(NgSpiceCommandError, "Loading circuit failed.");
        goto cleanup;
    }

cleanup:
    if (lines) free(lines);
    if (circuit_copy) free(circuit_copy);
    if (PyErr_Occurred()) return NULL;
    Py_RETURN_NONE;
}

static PyObject *shared_run(shared_t *self, PyObject *args, PyObject *kwds) {
    PyObject *background = Py_False;
    static char *kwlist[] = {"background", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &background))
        return NULL;

    char *command = background == Py_True ? "bg_run" : "run";
    PyObject_CallMethod((PyObject *)self, "exec_command", "s", command);

    if (background == Py_True) {
        self->is_running = true;
//        while (!no_bg) {
//            Py_BEGIN_ALLOW_THREADS
//            Py_END_ALLOW_THREADS
//        }
    }

    Py_RETURN_NONE;
}

static PyObject *shared_plot_names_getter(shared_t *self, void *closure) {
    char **plots = ngSpice_AllPlots();
    if (plots == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get plot names.");
        return NULL;
    }

    PyObject *list = PyList_New(0);
    for (int i = 0; plots[i] != NULL; i++) {
        PyObject *str = PyUnicode_FromString(plots[i]);
        if (!str || PyList_Append(list, str) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "plot_names append failed");
            Py_XDECREF(str);
            Py_DECREF(list);
            return NULL;
        }
        Py_DECREF(str);  // TODO: check!
    }
    return list;
}

static PyObject *shared_last_plot_getter(shared_t *self, void *closure) {
    char *plot = ngSpice_CurPlot();
    if (plot == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get last plot name.");
        return NULL;
    }

    PyObject *str = PyUnicode_FromString(plot);
    if (!str)
        return NULL;
    return str;
}

static PyObject *shared_plot(shared_t *self, PyObject *args) {
    char *plot_name;
    if (!PyArg_ParseTuple(args, "s", &plot_name))
        return NULL;

    char **all_vectors = ngSpice_AllVecs(plot_name);
    if (all_vectors == NULL) {
        PyErr_Format(PyExc_KeyError, "The plot `%s` does not exist.", plot_name);
        return NULL;
    }

    PyObject *plot = PyDict_New();
    if (!plot)
        return NULL;

    pvector_info vector_info;
    int length;
    for (int i = 0; all_vectors[i] != NULL; i++) {
        const char *vector_name = all_vectors[i];
        char *name = malloc(strlen(plot_name) + strlen(vector_name) + 2);
        sprintf(name, "%s.%s", plot_name, vector_name);

        vector_info = ngGet_Vec_Info(name);
        if (!vector_info) {
            PyErr_Format(NgSpiceCommandError, "Failed to get vector `%s` info.", name);
            return NULL;
        }
        free(name);

        length = vector_info->v_length;
        PyObject *np_array;
        npy_intp dims[1] = {length};
        if (vector_info->v_compdata == NULL) {
            np_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, vector_info->v_realdata);
            if (!np_array) {
                PyErr_SetString(PyExc_MemoryError, "Unable to create NumPy array.");
                return NULL;
            }
        } else {
            np_array = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
            if (!np_array) {
                PyErr_SetString(PyExc_MemoryError, "Unable to create NumPy array.");
                return NULL;
            }
            Py_complex *data = PyArray_DATA((PyArrayObject *)np_array);

            for (int j = 0; j < length; j++) {
                data[j].real = vector_info->v_compdata[j].cx_real;
                data[j].imag = vector_info->v_compdata[j].cx_imag;
            }
        }
        PyDict_SetItem(plot, PyUnicode_FromString(vector_name), np_array);
    }
    return plot;
}

static PyObject *shared_status(shared_t *self) {
    return PyObject_CallMethod((PyObject *)self, "exec_command", "s", "status");
}

static PyObject *shared_listing(shared_t *self) {
    return PyObject_CallMethod((PyObject *)self, "exec_command", "s", "listing");
}

static PyObject *shared_destroy(shared_t *self, PyObject *args, PyObject *kwds) {
    char *plot_name = "all";
    static char *kwlist[] = {"plot_name", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &plot_name))
        return NULL;

    char command[256];
    snprintf(command, sizeof(command), "destroy %s", plot_name);
    return PyObject_CallMethod((PyObject *)self, "exec_command", "s", command);
}

static PyObject *shared_remove_circuit(shared_t *self) {
    return PyObject_CallMethod((PyObject *)self, "exec_command", "s", "remcirc");
}

static PyObject *shared_reset(shared_t *self) {
    return PyObject_CallMethod((PyObject *)self, "exec_command", "s", "reset");
}

static void shared_dealloc(shared_t *self) {
    _string_array_free(&self->stdout_);
    _string_array_free(&self->stderr_);

    PyObject *instance_key = PyLong_FromLong(self->ngspice_id);
    PyDict_DelItem(instances_dict, instance_key);
    Py_DECREF(instance_key);

    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyMemberDef shared_members[] = {
    {"ngspice_id", T_INT, offsetof(shared_t, ngspice_id), READONLY, "ngspice_id"},
    {NULL}
};

static PyGetSetDef shared_getsetters[] = {
    {"stdout", (getter)shared_stdout_getter, NULL, "stdout string", NULL},
    {"stderr", (getter)shared_stderr_getter, NULL, "stderr string", NULL},
    {"_stdout", (getter)shared__stdout_getter, NULL, "stdout list", NULL},
    {"_stderr", (getter)shared__stderr_getter, NULL, "stderr list", NULL},
    {"plot_names", (getter)shared_plot_names_getter, NULL, "plot names", NULL},
    {"last_plot", (getter)shared_last_plot_getter, NULL, "last plot", NULL},
    {NULL}
};

static PyMethodDef shared_methods[] = {
    {"new_instance", (PyCFunction)shared_new_instance, METH_CLASS | METH_VARARGS | METH_KEYWORDS, "Create a new ngspice Shared instance"},
    {"_init_ngspice", (PyCFunction)shared__init_ngspice, METH_VARARGS | METH_KEYWORDS, "Initialize ngSpice with callbacks."},
    {"clear_output", (PyCFunction)shared_clear_output, METH_NOARGS, "Clear stdout and stderr"},
    {"load_circuit", (PyCFunction)shared_load_circuit, METH_VARARGS, "Load a circuit into ngSpice."},
    {"exec_command", (PyCFunction)shared_exec_command, METH_VARARGS | METH_KEYWORDS, "Execute a command in ngSpice."},
    {"run", (PyCFunction)shared_run, METH_VARARGS | METH_KEYWORDS, "Run the circuit."},
    {"status", (PyCFunction)shared_status, METH_NOARGS, "Get the status of the circuit."},
    {"listing", (PyCFunction)shared_listing, METH_NOARGS, "Get the listing of the circuit."},
    {"plot", (PyCFunction)shared_plot, METH_VARARGS, "Return the plot data of the circuit."},
    {"destroy", (PyCFunction)shared_destroy, METH_VARARGS | METH_KEYWORDS, "Destroy a plot."},
    {"remove_circuit", (PyCFunction)shared_remove_circuit, METH_NOARGS, "Remove the circuit."},
    {"reset", (PyCFunction)shared_reset, METH_NOARGS, "Reset the circuit."},
    {NULL}
};

static PyTypeObject shared_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_pyngspice.Shared",
    .tp_doc = "shared object",
    .tp_basicsize = sizeof(shared_t),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)shared___init__,
    .tp_dealloc = (destructor)shared_dealloc,
    .tp_methods = shared_methods,
    .tp_members = shared_members,
    .tp_getset = shared_getsetters,
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_pyngspice",
    .m_doc = "doc",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__pyngspice(void) {
    PyObject *m;

    if (PyType_Ready(&shared_type) < 0)
        return NULL;

    m = PyModule_Create(&moduledef);
    if (!m)
        goto cleanup;

    // Initialize NumPy
    import_array();

    if (PyModule_AddObject(m, "Shared", (PyObject *)&shared_type) < 0)
        goto cleanup;
    Py_INCREF(&shared_type);  // TODO: check!

    NgSpiceCommandError = PyErr_NewException("_pyngspice.NgSpiceCommandError", PyExc_RuntimeError, NULL);
    if (NgSpiceCommandError == NULL)
        goto cleanup;

    if (PyModule_AddObject(m, "NgSpiceCommandError", NgSpiceCommandError) < 0)
        goto cleanup;
    Py_INCREF(NgSpiceCommandError);

cleanup:
    if (PyErr_Occurred()) {
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}
