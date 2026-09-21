// Command server runs the unsaturated-infiltration HTTP service.
package main

import (
	"log"
	"os"

	"github.com/example/unsatflow/internal/api"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	s := api.New()
	log.Printf("unsatflow listening on :%s", port)
	if err := s.Router.Run(":" + port); err != nil {
		log.Fatal(err)
	}
}
